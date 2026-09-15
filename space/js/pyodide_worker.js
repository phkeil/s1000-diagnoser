// Web Worker: Pyodide runtime + the project's REAL preprocessing code.
//
// The whole point of this worker is train/serve parity. It does not
// reimplement the mel-spectrogram pipeline: it fetches the actual
// src/data.py, src/audio_utils.py, src/config.py and config/config.yaml
// (staged into space/python/ by scripts/deploy_space.sh) into Pyodide's
// virtual filesystem and imports them, so render_mel_spectrogram_image and
// chunk_audio here are literally the same functions src/train.py and
// api/main.py call.
//
// Getting librosa to import under Pyodide takes three shims, all of which are
// about IMPORTS ONLY - none of them touch the numbers:
//
//   * numba     - has no WebAssembly build and never will (LLVM JIT). librosa
//                 imports it at module load for @jit decorators. The mock
//                 supplies pass-through decorators; the mel path
//                 (filters.mel -> stft -> power_to_db) is pure numpy/scipy and
//                 never enters a jitted function, so the result is unchanged,
//                 just without the JIT speedup librosa would get natively.
//   * soundfile - needs libsndfile; librosa.core.audio imports it eagerly.
//   * audioread / pooch - same story, decode/download helpers we never reach.
//
// Audio never goes through librosa.load here (it cannot - no soundfile, and no
// m4a decoder in WASM). The main thread decodes with WebAudio and hands this
// worker raw mono float32 PCM at config.yaml's sample_rate. See app.js.
//
// Every fetch resolves against the baseUrl the main thread sends in `init`,
// never against this script's own location, so the worker works unchanged from
// space/index.html and from space/dev/parity.html.

const PYODIDE_VERSION = "0.27.0";
const PYODIDE_INDEX_URL = "https://cdn.jsdelivr.net/pyodide/v" + PYODIDE_VERSION + "/full/";

// Pinned to the exact version in requirements-api.txt / the training env.
// Parity is the reason this worker exists; a floating librosa would quietly
// undermine it.
const LIBROSA_VERSION = "0.10.2.post1";

// Stock Pyodide builds of everything librosa needs that DOES have a WASM
// build. Loading them up front keeps micropip from trying to fetch them from
// PyPI, where there is no wasm wheel to find.
const PYODIDE_PACKAGES = [
  "micropip", "numpy", "scipy", "matplotlib", "pillow", "pyyaml",
  "soxr", "scikit-learn", "joblib", "msgpack", "decorator", "lazy-loader",
  // sqlite3 is unvendored from Pyodide's stdlib and has to be asked for by
  // name: src/data.py imports src/manifest.py for its ManifestSegment type,
  // and that module opens the training manifest.
  "sqlite3",
];

// [path inside Pyodide's FS, path to fetch from, relative to baseUrl]
const PYTHON_SOURCES = [
  ["src/__init__.py", "python/src/__init__.py"],
  ["src/config.py", "python/src/config.py"],
  ["src/manifest.py", "python/src/manifest.py"],
  ["src/audio_utils.py", "python/src/audio_utils.py"],
  ["src/data.py", "python/src/data.py"],
  ["config/config.yaml", "python/config/config.yaml"],
];

const PYTHON_ROOT = "/s1000";

// Rendering a mel spectrogram through matplotlib is the slow step, so batches
// are posted as they finish: the main thread can start scoring batch 0 on the
// GPU while batch 1 is still rendering, and peak memory stays bounded instead
// of holding every segment of a long recording at once (a 60s upload is ~237
// segments = ~143 MB of float32 if accumulated).
const BATCH_SIZE = 32;

importScripts(PYODIDE_INDEX_URL + "pyodide.js");

let pyodide = null;
let baseUrl = null;

// Mock packages: satisfy librosa's dependency resolution and its eager imports
// without pulling in code that cannot exist in WASM. See the header comment.
const NUMBA_SHIM = `
def _passthrough(*args, **kwargs):
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]

    def wrap(func):
        return func

    return wrap


jit = njit = vectorize = guvectorize = stencil = _passthrough
prange = range
float32 = float64 = int32 = int64 = complex64 = complex128 = boolean = void = object()
`;

const UNAVAILABLE_SHIM = `
def __getattr__(name):
    # Dunder lookups must fail as a plain AttributeError: lazy_loader calls
    # inspect.stack(), which probes __file__ on every loaded module, and a
    # RuntimeError there aborts an unrelated import.
    if name.startswith("__"):
        raise AttributeError(name)
    raise RuntimeError(
        "%s is a stub under Pyodide. Audio decoding happens in the browser "
        "(WebAudio) before anything reaches Python." % __name__
    )
`;

// Runs inside Pyodide once src/ is importable. Only orchestration lives here:
// the mel rendering itself is src/data.py's, untouched.
const PY_GLUE = `
import json

import numpy as np
from PIL import Image
from matplotlib import colormaps
from matplotlib.colors import Normalize

from src.audio_utils import chunk_audio
from src.config import load_config
from src.data import render_mel_spectrogram_image

cfg = load_config()

# Same ImageNet constants src/data.py's build_transform() feeds to
# torchvision.transforms.Normalize. torchvision has no WASM build, but
# ToTensor + Normalize is exactly (uint8 / 255 - mean) / std followed by an
# HWC -> CHW transpose, which numpy reproduces bit for bit (the Resize in
# build_transform is a no-op: render_mel_spectrogram_image already returns
# cfg.image.size). Verified against torchvision by scripts/check_space_parity.py.
_MEAN = np.array(cfg.normalize.mean, dtype=np.float32)
_STD = np.array(cfg.normalize.std, dtype=np.float32)

# Matches api/inference_tab.py's cap on the full-file spectrogram.
MAX_SPECTROGRAM_WIDTH_PX = 4096

_state = {"audio": None, "sr": None, "chunks": []}


def config_json():
    """The config.yaml values the JS side needs, read out of the same Config
    object the preprocessing uses - so the UI and the scoring can never drift
    from the file that drove the rendering."""
    return json.dumps(
        {
            "audio": {
                "sample_rate": cfg.audio.sample_rate,
                "segment_duration": cfg.audio.segment_duration,
                "overlap": cfg.audio.overlap,
                "step_duration": cfg.audio.step_duration,
            },
            "image": {"size": list(cfg.image.size)},
            "anomaly": {
                "domain_baselines": dict(cfg.anomaly.domain_baselines),
                "default_domain": cfg.anomaly.default_domain,
                "garage_name_hints": list(cfg.anomaly.garage_name_hints),
                "rel_threshold": cfg.anomaly.rel_threshold,
                "window_size": cfg.anomaly.window_size,
                "healthy_median_score": cfg.anomaly.healthy_median_score,
            },
        }
    )


def prepare_audio(js_audio, sample_rate):
    """Store the decoded PCM and chunk it with the REAL chunk_audio(), so the
    segment grid is identical to the one src.inference.score_audio_file uses."""
    # np.array(..., copy=True) rather than asarray: to_py() hands back a
    # memoryview onto the JS heap, and an array that merely views it trips
    # numpy's stride bookkeeping inside librosa's as_strided framing. Owning
    # the buffer also detaches this from whatever JS does with the original.
    audio = np.array(js_audio.to_py(), dtype=np.float32, copy=True)
    audio = np.ascontiguousarray(audio)
    sample_rate = int(sample_rate)
    chunks = chunk_audio(audio, sample_rate, cfg.audio.segment_duration, cfg.audio.step_duration)

    _state["audio"] = audio
    _state["sr"] = sample_rate
    _state["chunks"] = chunks

    return json.dumps(
        {
            "num_segments": len(chunks),
            "duration": float(len(audio)) / float(sample_rate),
            "starts": [float(start) for start, _ in chunks],
        }
    )


def render_batch(lo, hi):
    """Segments [lo, hi) -> flat float32 bytes of shape (hi - lo, 3, H, W)."""
    tensors = []
    for _start, segment in _state["chunks"][int(lo):int(hi)]:
        image = render_mel_spectrogram_image(segment, _state["sr"], cfg)
        array = np.asarray(image, dtype=np.float32) / 255.0
        array = (array - _MEAN) / _STD
        tensors.append(np.ascontiguousarray(array.transpose(2, 0, 1), dtype=np.float32))

    return np.stack(tensors).astype(np.float32).tobytes()


def render_full_spectrogram():
    """Whole-file mel spectrogram, one pixel per hop, as PNG bytes.

    A deliberate mirror of api/inference_tab.py's
    _render_full_file_spectrogram_image (which is itself a documented copy of
    api/labeling.py's). It is NOT render_mel_spectrogram_image: that one always
    renders a fixed 224x224 figure for the model, whereas this view has to be
    one pixel per hop so the score graph can map time to x-position through the
    image's own width - the same geometry contract web/js/diagnose.js relies on.
    """
    import io

    import librosa

    mel_cfg = cfg.mel_spectrogram
    audio = _state["audio"]

    # Computed in blocks of frames rather than one melspectrogram() call, and
    # ONLY here - the per-segment path that feeds the model calls
    # render_mel_spectrogram_image unchanged.
    #
    # Why: librosa.util.frame() builds a strided view of shape
    # (len(y) - n_fft + 1, n_fft) before subsampling it by hop_length. numpy
    # rejects that view when size * itemsize overflows intp, and Pyodide is
    # wasm32, so intp maxes out at 2^31 - 1. At 44.1kHz / n_fft=2048 that caps
    # a single call at about 6 seconds of audio - fine for a 1s segment, not
    # for a whole recording.
    #
    # Blocking is exact, not an approximation: with center=True, frame t is
    # y_padded[t*hop : t*hop + n_fft], so a block of frames [t0, t1) is
    # recoverable from one center=False stft over that slice. Verified against
    # the single-call result at max abs diff == 0.0.
    FRAMES_PER_BLOCK = 128

    pad = mel_cfg.n_fft // 2
    hop = mel_cfg.hop_length
    # "constant" is librosa.stft's own default pad_mode.
    y_padded = np.pad(audio, pad, mode="constant")
    n_frames = 1 + len(audio) // hop

    blocks = []
    for t0 in range(0, n_frames, FRAMES_PER_BLOCK):
        t1 = min(t0 + FRAMES_PER_BLOCK, n_frames)
        segment = y_padded[t0 * hop : (t1 - 1) * hop + mel_cfg.n_fft]
        D = librosa.stft(segment, n_fft=mel_cfg.n_fft, hop_length=hop, center=False)
        blocks.append(np.abs(D) ** 2)

    power = np.concatenate(blocks, axis=1)
    S = librosa.feature.melspectrogram(
        S=power,
        sr=_state["sr"],
        n_fft=mel_cfg.n_fft,
        n_mels=mel_cfg.n_mels,
        fmin=mel_cfg.fmin,
        fmax=mel_cfg.fmax,
    )
    S_db = librosa.power_to_db(S, ref=np.max)

    norm = Normalize(vmin=mel_cfg.db_vmin, vmax=mel_cfg.db_vmax, clip=True)
    cmap = colormaps[cfg.image.colormap]
    # Flip rows so the lowest mel band renders at the bottom, matching
    # render_mel_spectrogram_image's imshow(..., origin="lower").
    rgb = (cmap(norm(S_db[::-1, :]))[:, :, :3] * 255).astype(np.uint8)
    image = Image.fromarray(rgb, mode="RGB")

    if image.width > MAX_SPECTROGRAM_WIDTH_PX:
        resample = getattr(Image.Resampling, cfg.image.resample)
        image = image.resize((MAX_SPECTROGRAM_WIDTH_PX, image.height), resample)

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def versions_json():
    import librosa
    import matplotlib
    import PIL
    import scipy
    import sys

    return json.dumps(
        {
            "python": sys.version.split()[0],
            "librosa": librosa.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
            "pillow": PIL.__version__,
        }
    )
`;

// Chained for the same reason as ort_worker.js: an async onmessage does not
// serialize messages, so a preprocess posted before init finished would run
// against a half-built runtime.
let queue = Promise.resolve();

self.onmessage = (event) => {
  queue = queue.then(() => handleMessage(event.data));
};

async function handleMessage(message) {
  try {
    if (message.type === "init") {
      await init(message.baseUrl);
    } else if (message.type === "preprocess") {
      await preprocess(message);
    }
  } catch (error) {
    self.postMessage({
      type: "error",
      stage: message && message.type,
      message: (error && error.message) || String(error),
    });
  }
}

function report(stage, detail) {
  self.postMessage({ type: "status", stage, detail });
}

async function fetchText(relativePath) {
  const url = new URL(relativePath, baseUrl).href;
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(
      "Could not fetch " + url + " (" + response.status + "). Run scripts/deploy_space.sh " +
      "to stage the Python sources into space/python/."
    );
  }
  return response.text();
}

async function init(mainThreadBaseUrl) {
  baseUrl = mainThreadBaseUrl;

  report("pyodide", "Loading Pyodide " + PYODIDE_VERSION + "…");
  pyodide = await loadPyodide({ indexURL: PYODIDE_INDEX_URL });

  report("pyodide", "Loading numpy, scipy, matplotlib…");
  await pyodide.loadPackage(PYODIDE_PACKAGES);

  report("pyodide", "Installing librosa " + LIBROSA_VERSION + "…");
  pyodide.globals.set("_numba_shim", NUMBA_SHIM);
  pyodide.globals.set("_unavailable_shim", UNAVAILABLE_SHIM);
  pyodide.globals.set("_librosa_version", LIBROSA_VERSION);
  await pyodide.runPythonAsync(`
import micropip

micropip.add_mock_package("numba", "0.61.0", modules={"numba": _numba_shim})
micropip.add_mock_package("soundfile", "0.12.1", modules={"soundfile": _unavailable_shim})
micropip.add_mock_package("audioread", "3.0.1", modules={"audioread": _unavailable_shim})
micropip.add_mock_package("pooch", "1.8.2", modules={"pooch": _unavailable_shim})

# torch/torchvision exist only because src/data.py imports them at module level
# for its Dataset classes and build_transform(). Nothing this worker calls uses
# them - the tensor conversion is done in numpy (see PY_GLUE).
micropip.add_mock_package(
    "torch",
    "2.11.0",
    modules={
        "torch": "class Tensor: pass",
        "torch.utils": "",
        "torch.utils.data": "class Dataset: pass",
    },
)
micropip.add_mock_package(
    "torchvision",
    "0.26.0",
    modules={
        "torchvision": "",
        "torchvision.transforms": (
            "class Compose:\\n"
            "    def __init__(self, transforms):\\n"
            "        self.transforms = transforms\\n"
            "class Resize:\\n"
            "    def __init__(self, *a, **k):\\n"
            "        pass\\n"
            "class ToTensor:\\n"
            "    def __init__(self, *a, **k):\\n"
            "        pass\\n"
            "class Normalize:\\n"
            "    def __init__(self, *a, **k):\\n"
            "        pass\\n"
        ),
    },
)

await micropip.install("librosa==" + _librosa_version)
`);

  report("pyodide", "Loading the project's preprocessing code…");
  const sources = await Promise.all(PYTHON_SOURCES.map(([, from]) => fetchText(from)));

  pyodide.FS.mkdirTree(PYTHON_ROOT + "/src");
  pyodide.FS.mkdirTree(PYTHON_ROOT + "/config");
  PYTHON_SOURCES.forEach(([target], index) => {
    pyodide.FS.writeFile(PYTHON_ROOT + "/" + target, sources[index]);
  });

  // src/config.py resolves config.yaml as __file__/../../config/config.yaml,
  // so this layout makes load_config() work with no arguments - exactly as it
  // does in training.
  pyodide.globals.set("_python_root", PYTHON_ROOT);
  await pyodide.runPythonAsync("import sys\nif _python_root not in sys.path:\n    sys.path.insert(0, _python_root)\n");

  await pyodide.runPythonAsync(PY_GLUE);

  const config = JSON.parse(callPython("config_json"));
  const versions = JSON.parse(callPython("versions_json"));

  self.postMessage({ type: "ready", config, versions });
}

// pyodide.globals.get() hands back a PyProxy that has to be freed explicitly;
// forgetting leaks the Python object for the life of the worker.
function callPython(name, ...args) {
  const fn = pyodide.globals.get(name);
  try {
    const result = fn(...args);
    if (result && typeof result.toJs === "function") {
      const value = result.toJs();
      result.destroy();
      return value;
    }
    return result;
  } finally {
    fn.destroy();
  }
}

async function preprocess(message) {
  const { audio, sampleRate } = message;

  const info = JSON.parse(callPython("prepare_audio", audio, sampleRate));
  if (info.num_segments === 0) {
    throw new Error(
      "Audio is shorter than the " + message.segmentDuration + "s analysis window - no segments to score."
    );
  }

  self.postMessage({
    type: "prepared",
    numSegments: info.num_segments,
    duration: info.duration,
    starts: info.starts,
  });

  // The full-file spectrogram first: it is what the UI can show immediately,
  // and it is one render rather than one per segment.
  const png = callPython("render_full_spectrogram");
  const pngBytes = png instanceof Uint8Array ? png : new Uint8Array(png);
  self.postMessage({ type: "spectrogram", png: pngBytes }, [pngBytes.buffer]);

  for (let lo = 0; lo < info.num_segments; lo += BATCH_SIZE) {
    const hi = Math.min(lo + BATCH_SIZE, info.num_segments);
    const bytes = callPython("render_batch", lo, hi);
    const u8 = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
    const data = new Float32Array(u8.buffer, u8.byteOffset, u8.byteLength / 4);

    self.postMessage(
      {
        type: "tensors",
        data,
        shape: [hi - lo, 3, message.imageSize[0], message.imageSize[1]],
        offset: lo,
        isLast: hi >= info.num_segments,
      },
      [data.buffer]
    );

    self.postMessage({ type: "progress", stage: "preprocess", done: hi, total: info.num_segments });
  }
}
