// Single wavesurfer instance, regions-as-labels: every region the user draws
// *is* a labeled stretch (no separate "segment" model underneath at labeling
// time - that only exists again server-side, at confirm, see api/labeling.py's
// confirm_upload). This module owns no label state of its own; it just wires
// wavesurfer/regions-plugin events through to upload.js's callbacks and
// exposes the handful of imperative actions (play, zoom, color, select,
// remove) upload.js needs to drive in response.
//
// No minimum region duration is enforced here (or anywhere client-side) -
// the backend already drops any region too short to yield a single
// chunk_audio window (see api/labeling.py's confirm_upload).

import WaveSurfer from "../vendor/wavesurfer.esm.js";
import RegionsPlugin from "../vendor/regions.esm.js";

export const LABEL_COLORS = {
  skip: "rgba(150, 150, 150, 0.3)",
  healthy: "rgba(34, 197, 94, 0.3)",
  defective: "rgba(239, 68, 68, 0.3)",
};

const SELECTED_OUTLINE = "2px solid #ffffff";

let wavesurfer = null;
let regionsPlugin = null;

let onRegionCreated = null;
let onRegionChanged = null;
let onRegionSettled = null;
let onRegionRemoved = null;
let onRegionClicked = null;
let onRegionDoubleClicked = null;
let onBackgroundClicked = null;
let onPlayStateChange = null;
let onPlaybackStateChange = null;
let onTimeUpdate = null;
let onScroll = null;
let onContentWidthChange = null;

// The one region currently playing (or null) - only one plays at a time.
let currentPlayingRegionId = null;

// The snap-guide line's DOM element, lazily created inside the current
// wrapper - reset to null on every initWaveform() so a fresh one gets
// created for the new wrapper (the old one is torn down with it).
let snapGuideElement = null;

export function initWaveform(
  container,
  {
    onRegionCreated: createdCallback,
    onRegionChanged: changedCallback,
    onRegionSettled: settledCallback,
    onRegionRemoved: removedCallback,
    onRegionClicked: clickedCallback,
    onRegionDoubleClicked: dblClickedCallback,
    onBackgroundClicked: backgroundClickedCallback,
    onPlayStateChange: playCallback,
    onPlaybackStateChange: playbackCallback,
    onTimeUpdate: timeUpdateCallback,
    onScroll: scrollCallback,
    onContentWidthChange: widthCallback,
  } = {}
) {
  if (wavesurfer) {
    wavesurfer.destroy();
  }
  onRegionCreated = createdCallback || null;
  onRegionChanged = changedCallback || null;
  onRegionSettled = settledCallback || null;
  onRegionRemoved = removedCallback || null;
  onRegionClicked = clickedCallback || null;
  onRegionDoubleClicked = dblClickedCallback || null;
  onBackgroundClicked = backgroundClickedCallback || null;
  onPlayStateChange = playCallback || null;
  onPlaybackStateChange = playbackCallback || null;
  onTimeUpdate = timeUpdateCallback || null;
  onScroll = scrollCallback || null;
  onContentWidthChange = widthCallback || null;
  currentPlayingRegionId = null;
  snapGuideElement = null;

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
  // timer), not just a full-track stop. The isPlaying() guard matters
  // because the browser's native 'pause' event is asynchronous: if the user
  // switches to a *different* region quickly, a stale 'pause' from the
  // previous stop() can arrive after the new region has already started,
  // and would otherwise wrongly clear the new region's playing state.
  wavesurfer.on("play", () => {
    onPlaybackStateChange?.(true);
  });
  wavesurfer.on("pause", () => {
    if (!wavesurfer.isPlaying()) {
      setPlayingRegion(null);
      onPlaybackStateChange?.(false);
    }
  });
  wavesurfer.on("finish", () => {
    if (!wavesurfer.isPlaying()) {
      setPlayingRegion(null);
      onPlaybackStateChange?.(false);
    }
  });
  wavesurfer.on("timeupdate", (currentTime) => {
    onTimeUpdate?.(currentTime);
  });

  wavesurfer.on("scroll", () => {
    onScroll?.(wavesurfer.getScroll());
  });
  wavesurfer.on("redrawcomplete", () => {
    onContentWidthChange?.(getContentWidthPx());
  });

  // Click-and-drag on empty waveform space draws a brand new region, styled
  // as "skip" (the same gray as an untouched/unlabeled region - see
  // web/js/segments.js's LABEL_TEXT) until the user assigns a real label.
  regionsPlugin.enableDragSelection({ color: LABEL_COLORS.skip, drag: true, resize: true });

  regionsPlugin.on("region-created", (region) => {
    onRegionCreated?.(region);
  });
  // "region-update" fires continuously while an *existing* region is being
  // dragged/resized (not during the initial creation-drag - the plugin only
  // starts listening to a region's own update events once it's saved, which
  // happens at the end of a creation-drag); "region-updated" fires once when
  // that drag/resize ends - upload.js uses the "settled" callback to run its
  // overlap check there, and the "changed" callback for live syncing.
  regionsPlugin.on("region-update", (region) => {
    onRegionChanged?.(region);
  });
  regionsPlugin.on("region-updated", (region) => {
    onRegionSettled?.(region);
  });
  regionsPlugin.on("region-removed", (region) => {
    if (currentPlayingRegionId === region.id) {
      setPlayingRegion(null);
    }
    onRegionRemoved?.(region.id);
  });
  regionsPlugin.on("region-clicked", (region) => {
    onRegionClicked?.(region);
  });
  regionsPlugin.on("region-double-clicked", (region) => {
    onRegionDoubleClicked?.(region);
  });

  // A plain click that does NOT land on any region's own DOM element clears
  // the current selection - checked against each region's real .element
  // (not an attribute selector, which the drag-select path never sets on
  // time). wavesurfer's own default click-to-seek behavior is left
  // completely alone; this listener is purely additive.
  wavesurfer.getWrapper().addEventListener("click", (event) => {
    if (!isClickOnRegion(event.target)) {
      onBackgroundClicked?.();
    }
  });

  return wavesurfer;
}

function isClickOnRegion(target) {
  return regionsPlugin.getRegions().some((region) => region.element && region.element.contains(target));
}

function setPlayingRegion(regionId) {
  currentPlayingRegionId = regionId;
  onPlayStateChange?.(regionId);
}

export async function loadFileIntoWaveform(file) {
  await wavesurfer.loadBlob(file);
  onContentWidthChange?.(getContentWidthPx());
}

export function getDuration() {
  return wavesurfer?.getDuration() || 0;
}

// The waveform's current rendered content width in pixels - "100%" of the
// panel when zoomed out to fit, or wider once zoomed in. Reported through
// onContentWidthChange so upload.js can keep the spectrogram image at this
// same CSS width - that's what makes a shared scrollLeft line up in both.
function getContentWidthPx() {
  return wavesurfer?.getWrapper()?.clientWidth || 0;
}

// minPxPerSec: 0 means "fit the full file to the container width" (wavesurfer's
// own default with no zoom applied).
export function zoomTo(minPxPerSec) {
  if (!wavesurfer || !wavesurfer.getDuration()) {
    return;
  }
  wavesurfer.zoom(minPxPerSec);
}

export function setScroll(scrollLeftPx) {
  wavesurfer?.setScroll(scrollLeftPx);
}

// --- Global transport (plays/pauses the shared cursor position, independent
// of any single region) ---

export function play() {
  wavesurfer?.play();
}

export function pause() {
  wavesurfer?.pause();
}

export function seekToStart() {
  if (!wavesurfer) {
    return;
  }
  wavesurfer.seekTo(0);
  wavesurfer.pause();
}

// Toggles: playing the region that's already playing stops it; playing a
// different one just starts it (play() reseeks/restarts on its own), so
// only one is ever playing.
export function playRegion(region) {
  if (!wavesurfer) {
    return;
  }
  if (currentPlayingRegionId === region.id) {
    wavesurfer.stop();
    return;
  }
  setPlayingRegion(region.id);
  wavesurfer.play(region.start, region.end);
}

export function stopPlayback() {
  wavesurfer?.stop();
}

export function removeRegion(region) {
  region.remove();
}

export function setRegionLabel(region, label) {
  region.setOptions({ color: LABEL_COLORS[label] });
}

export function setRegionSelected(region, isSelected) {
  if (!region.element) {
    return;
  }
  region.element.style.outline = isSelected ? SELECTED_OUTLINE : "";
  region.element.style.zIndex = isSelected ? "3" : "";
}

// Used by upload.js's overlap check to revert a region back to its last
// known-good {start, end} after a rejected drag/resize.
export function setRegionPosition(region, { start, end }) {
  region.setOptions({ start, end });
}

// Brings a region's start time into view (at the viewport's left edge) -
// useful once zoomed in far enough that a region drawn/selected elsewhere
// scrolls off-screen.
export function scrollToRegion(region) {
  wavesurfer?.setScrollTime(region.start);
}

// Thin vertical guide line at a candidate snap position, live during a
// drag/resize - positioned as a percentage of duration inside wavesurfer's
// own (zoom-aware) wrapper, the same coordinate space regions themselves
// use, so it stays aligned at any zoom/scroll position.
export function showSnapGuide(timeSeconds) {
  const duration = wavesurfer?.getDuration();
  if (!duration) {
    return;
  }
  if (!snapGuideElement) {
    snapGuideElement = document.createElement("div");
    snapGuideElement.className = "snap-guide";
    wavesurfer.getWrapper().appendChild(snapGuideElement);
  }
  snapGuideElement.style.left = `${(timeSeconds / duration) * 100}%`;
  snapGuideElement.hidden = false;
}

export function hideSnapGuide() {
  if (snapGuideElement) {
    snapGuideElement.hidden = true;
  }
}
