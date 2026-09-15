// Orchestration: file -> decoded PCM -> pyodide_worker (mel tensors) ->
// ort_worker (reconstruction MSE + scoring) -> ui.
//
// Nothing here computes anything about the audio itself. The numbers come from
// the project's own Python (in the Pyodide worker) and from the exported CAE
// (in the ORT worker); this module only moves data between them and decides
// which domain baseline applies.
//
// The two workers boot in parallel and are both slow (Pyodide ~10s, the 110 MB
// model ~5-30s depending on connection), so they start on page load rather
// than on first upload.

import * as ui from "./ui.js";

const MAX_UPLOAD_BYTES = 50 * 1024 * 1024; // same cap as api/main.py
const ALLOWED_SUFFIXES = [".wav", ".m4a"];
const THRESHOLDS_PATH = "assets/thresholds.json";

const baseUrl = new URL(".", window.location.href).href;

const pyodideWorker = new Worker(new URL("./pyodide_worker.js", import.meta.url));
const ortWorker = new Worker(new URL("./ort_worker.js", import.meta.url));

const state = {
  config: null, // config.yaml, as read by Pyodide's load_config()
  thresholds: null, // assets/thresholds.json, if deployed alongside the model
  versions: null,
  provider: null,
  pyodideReady: false,
  ortReady: false,
  busy: false,
  run: null,
};

// ---------------------------------------------------------------------------
// Worker plumbing
// ---------------------------------------------------------------------------

pyodideWorker.onmessage = (event) => {
  const message = event.data;
  switch (message.type) {
    case "status":
      ui.setBootStatus("pyodide", message.detail);
      break;
    case "ready":
      state.config = message.config;
      state.versions = message.versions;
      state.pyodideReady = true;
      ui.setBootStatus("pyodide", "Python ready (librosa " + message.versions.librosa + ")");
      onWorkerReady();
      break;
    case "prepared":
      state.run.numSegments = message.numSegments;
      state.run.duration = message.duration;
      state.run.starts = message.starts;
      ui.setAnalysisStatus("Rendering " + message.numSegments + " mel spectrograms…");
      ui.beginResults({ duration: message.duration, domain: state.run.domain, filename: state.run.filename });
      break;
    case "spectrogram":
      ui.setSpectrogram(message.png);
      break;
    case "tensors":
      // Forward straight to the ORT worker; the buffer is transferred, so this
      // never copies 18 MB of float32 per batch.
      ortWorker.postMessage({ type: "score", data: message.data, offset: message.offset }, [message.data.buffer]);
      if (message.isLast) {
        state.run.allBatchesSent = true;
        maybeFinalize();
      }
      break;
    case "progress":
      ui.setProgress("Preprocessing", message.done, message.total);
      break;
    case "error":
      failRun(message.message);
      break;
    default:
      break;
  }
};

ortWorker.onmessage = (event) => {
  const message = event.data;
  switch (message.type) {
    case "ready":
      state.ortReady = true;
      state.provider = message.provider;
      ui.setBootStatus("ort", "Model ready (" + message.provider + ")");
      onWorkerReady();
      break;
    case "progress":
      state.run.scored = message.done;
      ui.setProgress("Scoring", message.done, state.run.numSegments);
      maybeFinalize();
      break;
    case "results":
      finishRun(message);
      break;
    case "error":
      failRun(message.message);
      break;
    default:
      break;
  }
};

function onWorkerReady() {
  if (state.pyodideReady && state.ortReady) {
    ui.setReady(state.versions, state.provider);
  }
}

// The ORT worker has no idea how many segments are coming, so the main thread
// decides when everything has been scored: all batches handed over AND every
// segment accounted for.
function maybeFinalize() {
  const run = state.run;
  if (!run || run.finalized || !run.allBatchesSent) {
    return;
  }
  if (run.scored < run.numSegments) {
    return;
  }
  run.finalized = true;

  ortWorker.postMessage({
    type: "finalize",
    starts: run.starts,
    duration: run.duration,
    segmentDuration: state.config.audio.segment_duration,
    params: run.params,
  });
}

// ---------------------------------------------------------------------------
// Thresholds: mirrors api/main.py + src/thresholds.py's apply_thresholds
// ---------------------------------------------------------------------------

// A model trained by src/train.py gets its own val-set-derived thresholds.
// If assets/thresholds.json was not deployed (e.g. a pre-Phase-1 checkpoint),
// config.yaml's hardcoded anomaly values are used unchanged - the same
// graceful degradation the server does.
async function loadThresholds() {
  try {
    const response = await fetch(new URL(THRESHOLDS_PATH, baseUrl).href);
    if (!response.ok) {
      return null;
    }
    const data = await response.json();
    if (typeof data.rel_threshold !== "number" || typeof data.healthy_median_score !== "number") {
      console.warn("Ignoring malformed thresholds.json");
      return null;
    }
    return data;
  } catch (error) {
    return null;
  }
}

// domain_baselines is MERGED over config.yaml's rather than replacing it, so a
// domain absent from the run's val set keeps its fallback (apply_thresholds).
function resolveScoringParams(domain) {
  const anomaly = state.config.anomaly;
  const thresholds = state.thresholds;

  const baselines = { ...anomaly.domain_baselines, ...((thresholds && thresholds.domain_baselines) || {}) };

  return {
    domain,
    baseline: baselines[domain],
    threshold: thresholds ? thresholds.rel_threshold : anomaly.rel_threshold,
    healthyMedianScore: thresholds ? thresholds.healthy_median_score : anomaly.healthy_median_score,
    windowSize: anomaly.window_size,
  };
}

// src/inference.py's infer_domain(), with config.yaml's own hint list.
function inferDomain(filename) {
  const anomaly = state.config.anomaly;
  return anomaly.garage_name_hints.some((hint) => filename.includes(hint)) ? "Garage" : anomaly.default_domain;
}

// ---------------------------------------------------------------------------
// Audio decoding
// ---------------------------------------------------------------------------

// Decoded here rather than with librosa.load in Pyodide, which cannot work:
// soundfile needs libsndfile and there is no m4a decoder in WASM. The browser
// already ships one, and OfflineAudioContext resamples to config.yaml's
// sample_rate as part of decoding - so a 44.1kHz WAV round-trips with no
// resampling at all, and anything else is resampled by the browser instead of
// by soxr. Multi-channel input is averaged to mono, matching librosa's
// to_mono(). This is the one place the browser pipeline is not running the
// project's own Python.
async function decodeAudio(file, sampleRate) {
  const bytes = await file.arrayBuffer();
  const context = new OfflineAudioContext(1, 1, sampleRate);
  const buffer = await context.decodeAudioData(bytes);

  if (buffer.numberOfChannels === 1) {
    return buffer.getChannelData(0);
  }

  const mono = new Float32Array(buffer.length);
  for (let channel = 0; channel < buffer.numberOfChannels; channel += 1) {
    const channelData = buffer.getChannelData(channel);
    for (let i = 0; i < channelData.length; i += 1) {
      mono[i] += channelData[i];
    }
  }
  for (let i = 0; i < mono.length; i += 1) {
    mono[i] /= buffer.numberOfChannels;
  }
  return mono;
}

// ---------------------------------------------------------------------------
// Run lifecycle
// ---------------------------------------------------------------------------

async function analyze(file, domainChoice) {
  if (state.busy) {
    return;
  }

  const suffix = file.name.slice(file.name.lastIndexOf(".")).toLowerCase();
  if (!ALLOWED_SUFFIXES.includes(suffix)) {
    ui.setAnalysisStatus("Unsupported file type '" + suffix + "'. Expected one of " + ALLOWED_SUFFIXES.join(", ") + ".", true);
    return;
  }
  if (file.size > MAX_UPLOAD_BYTES) {
    ui.setAnalysisStatus("File too large. Maximum size is 50 MB.", true);
    return;
  }

  state.busy = true;
  ui.setBusy(true);
  ui.resetResults();

  const domain = domainChoice || inferDomain(file.name);
  state.run = {
    filename: file.name,
    domain,
    params: resolveScoringParams(domain),
    numSegments: 0,
    scored: 0,
    starts: [],
    duration: 0,
    allBatchesSent: false,
    finalized: false,
    startedAt: performance.now(),
  };

  if (state.run.params.baseline === undefined) {
    failRun("No healthy baseline configured for domain '" + domain + "'.");
    return;
  }

  try {
    ui.setAnalysisStatus("Decoding audio…");
    const audio = await decodeAudio(file, state.config.audio.sample_rate);

    ortWorker.postMessage({ type: "reset" });
    pyodideWorker.postMessage(
      {
        type: "preprocess",
        audio,
        sampleRate: state.config.audio.sample_rate,
        segmentDuration: state.config.audio.segment_duration,
        imageSize: state.config.image.size,
      },
      [audio.buffer]
    );
  } catch (error) {
    failRun("Could not decode audio file: " + ((error && error.message) || error));
  }
}

function finishRun(results) {
  const elapsed = (performance.now() - state.run.startedAt) / 1000;
  ui.renderResults(results);
  ui.setAnalysisStatus(
    "Analyzed \"" + state.run.filename + "\" - " + results.segments.length + " segments in " +
    elapsed.toFixed(1) + "s (" + results.provider + ", " + Math.round(results.inference_ms) + "ms of inference)."
  );
  state.busy = false;
  ui.setBusy(false);
}

function failRun(reason) {
  ui.setAnalysisStatus(reason, true);
  ui.resetResults();
  state.busy = false;
  ui.setBusy(false);
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

ui.init({ onAnalyze: analyze });

loadThresholds().then((thresholds) => {
  state.thresholds = thresholds;
  ui.setThresholdSource(thresholds);
});

pyodideWorker.postMessage({ type: "init", baseUrl });
ortWorker.postMessage({ type: "init", baseUrl });
