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
  boot: {
    loaded: 0, // model bytes received
    total: 0, // model bytes expected; 0 until content-length is known
    downloaded: false,
    toolsStep: 0, // which of the Pyodide worker's boot steps is running
    toolsSteps: 0,
  },
};

// The two workers boot in parallel, so the bar is a weighted sum of both
// rather than either one's own progress. The weights are roughly what each
// part costs on a cold load: the 110 MB download dominates, building the
// session is a few seconds, and the Python runtime comes up alongside them.
const BOOT_WEIGHTS = { download: 0.7, session: 0.1, tools: 0.2 };

const BYTES_PER_MB = 1024 * 1024;

// ---------------------------------------------------------------------------
// Worker plumbing
// ---------------------------------------------------------------------------

pyodideWorker.onmessage = (event) => {
  const message = event.data;
  switch (message.type) {
    case "status":
      // The worker's own wording names Pyodide and librosa, which the page
      // deliberately does not; only its position in the sequence is used.
      console.info("[pyodide]", message.detail);
      state.boot.toolsStep = message.step || 0;
      state.boot.toolsSteps = message.totalSteps || 0;
      updateBootProgress();
      break;
    case "ready":
      state.config = message.config;
      state.versions = message.versions;
      state.pyodideReady = true;
      updateBootProgress();
      onWorkerReady();
      break;
    case "prepared":
      state.run.numSegments = message.numSegments;
      state.run.duration = message.duration;
      state.run.starts = message.starts;
      ui.setAnalysisStatus("Analyzing " + message.numSegments + " sections…");
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
      ui.setProgress("Analyzing", message.done, message.total);
      break;
    case "error":
      reportError(message.message);
      break;
    default:
      break;
  }
};

ortWorker.onmessage = (event) => {
  const message = event.data;
  switch (message.type) {
    case "download":
      state.boot.loaded = message.loaded;
      state.boot.total = message.total;
      updateBootProgress();
      break;
    case "downloaded":
      state.boot.downloaded = true;
      updateBootProgress();
      break;
    case "ready":
      state.ortReady = true;
      state.provider = message.provider;
      updateBootProgress();
      onWorkerReady();
      break;
    case "progress":
      state.run.scored = message.done;
      ui.setProgress("Running the diagnosis", message.done, state.run.numSegments);
      maybeFinalize();
      break;
    case "results":
      finishRun(message);
      break;
    case "error":
      reportError(message.message);
      break;
    default:
      break;
  }
};

function onWorkerReady() {
  if (state.pyodideReady && state.ortReady) {
    ui.setReady();
  }
}

// A step that is RUNNING is not a step that is done, so the Pyodide side
// contributes (step - 1) / steps and only reaches its full weight on ready.
// That keeps the bar from sitting at 100% while the last step still works.
function toolsFraction() {
  const { toolsStep, toolsSteps } = state.boot;
  if (state.pyodideReady) {
    return 1;
  }
  return toolsSteps ? Math.max(0, toolsStep - 1) / toolsSteps : 0;
}

function downloadFraction() {
  const { loaded, total, downloaded } = state.boot;
  if (downloaded) {
    return 1;
  }
  return total ? Math.min(1, loaded / total) : null;
}

function formatMb(bytes) {
  return Math.round(bytes / BYTES_PER_MB) + " MB";
}

function updateBootProgress() {
  if (state.pyodideReady && state.ortReady) {
    return;
  }

  const download = downloadFraction();
  const fraction =
    download === null
      ? null
      : BOOT_WEIGHTS.download * download +
        BOOT_WEIGHTS.session * (state.ortReady ? 1 : 0) +
        BOOT_WEIGHTS.tools * toolsFraction();

  let message = "Getting ready…";
  let detail = "This takes a moment the first time.";

  if (!state.boot.downloaded && state.boot.loaded > 0) {
    message = "Downloading the analyzer…";
    detail = state.boot.total
      ? formatMb(state.boot.loaded) + " of " + formatMb(state.boot.total)
      : formatMb(state.boot.loaded) + " so far";
  } else if (state.boot.downloaded && !state.ortReady) {
    message = "Setting things up…";
    detail = "Almost there.";
  } else if (state.boot.downloaded) {
    message = "Almost ready…";
    detail = "";
  }

  ui.setBootProgress({ fraction, message, detail });
}

// A worker that dies before both are up means there is nothing to upload into,
// so the page says so once instead of surfacing the worker's own message.
function reportError(message) {
  if (!state.pyodideReady || !state.ortReady) {
    console.error("Analyzer failed to start:", message);
    ui.setBootError();
    return;
  }
  failRun(message);
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
    ui.setAnalysisStatus("That file type can't be read. Please upload a .wav or .m4a recording.", true);
    return;
  }
  if (file.size > MAX_UPLOAD_BYTES) {
    ui.setAnalysisStatus("That recording is too long. Try a clip of about 15-30 seconds.", true);
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
    console.error("No healthy baseline configured for domain '" + domain + "'.");
    failRun("Something went wrong setting up the analysis. Try refreshing the page.");
    return;
  }

  try {
    ui.setAnalysisStatus("Reading your recording…");
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
    console.error("Could not decode audio file:", error);
    failRun("That recording couldn't be read. Try a different file.");
  }
}

// The timings and the execution provider are for the console, not the page:
// the result the rider needs is the verdict, which ui.renderResults leads with.
function finishRun(results) {
  const elapsed = (performance.now() - state.run.startedAt) / 1000;
  console.info(
    "Analyzed \"" + state.run.filename + "\" - " + results.segments.length + " segments in " +
    elapsed.toFixed(1) + "s (" + results.provider + ", " + Math.round(results.inference_ms) + "ms of inference)."
  );
  ui.renderResults(results);
  ui.setAnalysisStatus("Done, your result is below.");
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
});

pyodideWorker.postMessage({ type: "init", baseUrl });
ortWorker.postMessage({ type: "init", baseUrl });
