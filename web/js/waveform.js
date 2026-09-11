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

// The one segment currently playing (or null) - only one plays at a time,
// see playSegment() below.
let currentPlayingSegmentId = null;
let onPlayStateChange = null;

export function initWaveform(container, { onPlayStateChange: callback } = {}) {
  if (wavesurfer) {
    wavesurfer.destroy();
  }
  onPlayStateChange = callback || null;
  currentPlayingSegmentId = null;

  regionsPlugin = RegionsPlugin.create();
  wavesurfer = WaveSurfer.create({
    container,
    waveColor: "#5b8def",
    progressColor: "#3a6bd1",
    height: 96,
    plugins: [regionsPlugin],
  });

  // wavesurfer's own pause/finish events also fire for a region-limited
  // play(start, end) reaching its end (via its internal stopAtPosition
  // timer), not just a full-track stop - that's what lets us detect
  // "playback actually stopped" without polling. The isPlaying() guard
  // matters because the browser's native 'pause' event is asynchronous: if
  // the user switches to a *different* segment quickly, a stale 'pause' from
  // the previous stop() can arrive after the new segment has already
  // started, and would otherwise wrongly clear the new segment's state.
  wavesurfer.on("pause", () => {
    if (!wavesurfer.isPlaying()) {
      setPlayingSegment(null);
    }
  });
  wavesurfer.on("finish", () => {
    if (!wavesurfer.isPlaying()) {
      setPlayingSegment(null);
    }
  });

  regionsBySegmentId = new Map();
  return wavesurfer;
}

function setPlayingSegment(segmentId) {
  currentPlayingSegmentId = segmentId;
  onPlayStateChange?.(segmentId);
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

// Toggles: clicking the segment that's already playing stops it; clicking a
// different one stops nothing explicitly (play() reseeks/restarts on its
// own) and just starts the new one, so only one is ever playing.
export function playSegment(segmentId, startTime, endTime) {
  if (!wavesurfer) {
    return;
  }
  if (currentPlayingSegmentId === segmentId) {
    wavesurfer.stop();
    return;
  }
  setPlayingSegment(segmentId);
  wavesurfer.play(startTime, endTime);
}

export function stopPlayback() {
  wavesurfer?.stop();
}

export function playFromStart() {
  if (!wavesurfer) {
    return;
  }
  wavesurfer.stop();
  // setTime() (unlike stop() alone) explicitly clears any stopAtPosition
  // left over from the last per-segment play(start, end) - without it, a
  // "play from start" right after playing a segment can stop early at that
  // segment's old end time instead of playing the full track.
  wavesurfer.setTime(0);
  wavesurfer.play();
}
