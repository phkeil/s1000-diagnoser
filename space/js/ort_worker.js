// Web Worker: ONNX Runtime Web (WebGPU, WASM fallback) + anomaly scoring.
//
// Two responsibilities, in the same order src.inference does them:
//
//   1. reconstruction MSE per segment - the CAE is trained on healthy audio
//      only, so how badly it reconstructs a segment IS the anomaly signal.
//   2. the post-processing in src/inference.py's score_segments(): normalize
//      by the domain's healthy baseline, smooth with a centered rolling mean,
//      compare against rel_threshold, and turn the distance from that
//      threshold into a confidence with anomaly_confidence()'s sigmoid.
//
// Step 2 is a line-by-line port and has to stay one. Anything that drifts here
// - the rolling window's alignment, the sigmoid's scale - changes the verdict
// for the same audio the server would call healthy. The parity script
// (scripts/check_space_parity.py) exists to catch exactly that.
//
// Segments are scored one at a time rather than as one N-batch: a long
// recording is hundreds of segments, and a single (N, 3, 224, 224) tensor is
// both a large allocation and a long GPU stall. The exported graph has a
// dynamic batch axis, so batching remains possible if that ever changes.

importScripts("https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js");

const MODEL_PATH = "assets/s1000_cae.onnx";
const INPUT_NAME = "input";
const OUTPUT_NAME = "reconstruction";
const SEGMENT_FLOATS = 3 * 224 * 224;

// src/inference.py's _CONFIDENCE_FLOOR: a typical healthy segment should read
// as low-but-not-zero confidence rather than flatlining at 0.
const CONFIDENCE_FLOOR = 0.05;

// Without this, ORT looks for its .wasm/.mjs runtime next to this worker.
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";

let session = null;
let provider = null;
let rawScores = [];
let inferenceMs = 0;

// Messages are chained rather than handled directly, because an async
// onmessage does NOT serialize them: the worker dispatches the next message as
// soon as the previous handler hits an await, so a "finalize" posted right
// after the last "score" would run while that score was still inside
// session.run() - and score an empty array. Queuing makes ordering a property
// of this worker instead of something every caller has to arrange.
let queue = Promise.resolve();

self.onmessage = (event) => {
  queue = queue.then(() => handleMessage(event.data));
};

async function handleMessage(message) {
  try {
    if (message.type === "init") {
      await init(message.baseUrl);
    } else if (message.type === "reset") {
      rawScores = [];
      inferenceMs = 0;
    } else if (message.type === "score") {
      await score(message);
    } else if (message.type === "finalize") {
      finalize(message);
    }
  } catch (error) {
    self.postMessage({
      type: "error",
      stage: message && message.type,
      message: (error && error.message) || String(error),
    });
  }
}

async function init(baseUrl) {
  const modelUrl = new URL(MODEL_PATH, baseUrl).href;

  // navigator.gpu exists in a worker on Chrome/Edge; the catch also covers a
  // browser that advertises WebGPU but cannot build a session on it.
  if (self.navigator && self.navigator.gpu) {
    try {
      session = await ort.InferenceSession.create(modelUrl, { executionProviders: ["webgpu"] });
      provider = "WebGPU";
    } catch (error) {
      console.warn("WebGPU session creation failed, falling back to WASM:", error);
    }
  }

  if (!session) {
    session = await ort.InferenceSession.create(modelUrl, { executionProviders: ["wasm"] });
    provider = self.navigator && self.navigator.gpu ? "WASM (WebGPU unavailable)" : "WASM (no WebGPU in this browser)";
  }

  self.postMessage({ type: "ready", provider });
}

async function score(message) {
  const { data, offset } = message;
  const count = data.length / SEGMENT_FLOATS;

  for (let i = 0; i < count; i += 1) {
    const input = data.subarray(i * SEGMENT_FLOATS, (i + 1) * SEGMENT_FLOATS);
    const tensor = new ort.Tensor("float32", input, [1, 3, 224, 224]);

    const started = performance.now();
    const outputs = await session.run({ [INPUT_NAME]: tensor });
    const output = outputs[OUTPUT_NAME] || Object.values(outputs)[0];
    // A WebGPU-resident output has to be pulled back to the CPU explicitly.
    const reconstruction =
      output.location && output.location !== "cpu" && typeof output.getData === "function"
        ? await output.getData()
        : output.data;
    inferenceMs += performance.now() - started;

    // torch.nn.functional.mse_loss(reconstruction, input): mean of the squared
    // difference over every element.
    let sum = 0;
    for (let j = 0; j < input.length; j += 1) {
      const diff = reconstruction[j] - input[j];
      sum += diff * diff;
    }
    rawScores[offset + i] = sum / input.length;
  }

  self.postMessage({ type: "progress", stage: "score", done: offset + count });
}

// ---------------------------------------------------------------------------
// Port of src/inference.py's score_segments() / anomaly_confidence()
// ---------------------------------------------------------------------------

// pandas .rolling(window=w, center=True, min_periods=1).mean(). For an even
// window pandas leans BACKWARDS: index i averages [i - w//2, i + (w-1)//2],
// clipped at both ends (min_periods=1 means short windows still produce a
// value rather than NaN). Verified against pandas for w=4 - getting this
// off by one would shift every score by up to half a window in time.
function rollingMeanCentered(values, windowSize) {
  const n = values.length;
  const back = Math.floor(windowSize / 2);
  const forward = Math.floor((windowSize - 1) / 2);

  const out = new Array(n);
  for (let i = 0; i < n; i += 1) {
    const lo = Math.max(i - back, 0);
    const hi = Math.min(i + forward, n - 1);
    let sum = 0;
    for (let j = lo; j <= hi; j += 1) {
      sum += values[j];
    }
    out[i] = sum / (hi - lo + 1);
  }
  return out;
}

// src/inference.py's _stable_sigmoid: math.exp() only ever sees a
// non-positive argument, so a pathological threshold cannot overflow.
function stableSigmoid(x) {
  if (x >= 0) {
    return 1 / (1 + Math.exp(-x));
  }
  const expX = Math.exp(x);
  return expX / (1 + expX);
}

function anomalyConfidence(smoothedScore, threshold, healthyMedianScore) {
  const spread = Math.max(threshold - healthyMedianScore, 1e-6);
  const scale = spread / Math.log((1 - CONFIDENCE_FLOOR) / CONFIDENCE_FLOOR);
  return stableSigmoid((smoothedScore - threshold) / scale);
}

function finalize(message) {
  const { starts, duration, segmentDuration, params } = message;
  const { baseline, threshold, windowSize, healthyMedianScore, domain } = params;

  const relative = rawScores.map((raw) => raw / baseline);
  const smoothed = rollingMeanCentered(relative, windowSize);

  const segments = rawScores.map((raw, index) => ({
    index,
    start_time: starts[index],
    end_time: starts[index] + segmentDuration,
    raw_mse: raw,
    relative_score: relative[index],
    // The SMOOTHED score, i.e. exactly the quantity compared against the
    // threshold - so the graph and is_anomalous can never disagree.
    anomaly_score: smoothed[index],
    is_anomalous: smoothed[index] > threshold,
    confidence: anomalyConfidence(smoothed[index], threshold, healthyMedianScore),
  }));

  const isAnomalous = segments.some((segment) => segment.is_anomalous);
  const overallConfidence = segments.reduce((max, segment) => Math.max(max, segment.confidence), 0);

  self.postMessage({
    type: "results",
    provider,
    domain,
    duration_seconds: duration,
    threshold,
    baseline,
    overall_verdict: isAnomalous ? "anomaly" : "healthy",
    overall_confidence: overallConfidence,
    segments,
    inference_ms: inferenceMs,
  });
}
