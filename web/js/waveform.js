// Single wavesurfer instance: loads the original uploaded File client-side
// (no backend audio-serving endpoint - see plan reconciliation notes) and
// draws one colored Region per chunk, doubling as a visual overview of
// labeling progress.

import WaveSurfer from "../vendor/wavesurfer.esm.js";
import RegionsPlugin from "../vendor/regions.esm.js";

const REGION_COLOR = {
  unlabeled: "rgba(107, 114, 128, 0.35)",
  healthy: "rgba(61, 220, 132, 0.35)",
  defective: "rgba(242, 84, 91, 0.35)",
};

let wavesurfer = null;
let regionsPlugin = null;
let regionsBySegmentId = new Map();

export function initWaveform(container) {
  if (wavesurfer) {
    wavesurfer.destroy();
  }
  regionsPlugin = RegionsPlugin.create();
  wavesurfer = WaveSurfer.create({
    container,
    waveColor: "#5b8def",
    progressColor: "#3a6bd1",
    height: 96,
    plugins: [regionsPlugin],
  });
  regionsBySegmentId = new Map();
  return wavesurfer;
}

export async function loadFileIntoWaveform(file) {
  await wavesurfer.loadBlob(file);
}

// segments: [{segmentId, startTime, endTime, label}]
export function drawSegmentRegions(segments) {
  regionsPlugin.clearRegions();
  regionsBySegmentId = new Map();
  for (const segment of segments) {
    const region = regionsPlugin.addRegion({
      start: segment.startTime,
      end: segment.endTime,
      color: REGION_COLOR[segment.label] || REGION_COLOR.unlabeled,
      drag: false,
      resize: false,
    });
    regionsBySegmentId.set(segment.segmentId, region);
  }
}

export function setRegionLabel(segmentId, label) {
  const region = regionsBySegmentId.get(segmentId);
  if (region) {
    region.setOptions({ color: REGION_COLOR[label] || REGION_COLOR.unlabeled });
  }
}

export function playSegment(startTime, endTime) {
  wavesurfer?.play(startTime, endTime);
}

export function playFromStart() {
  if (!wavesurfer) {
    return;
  }
  wavesurfer.stop();
  wavesurfer.play();
}
