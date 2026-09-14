// Upload form + region-based labeling: owns every piece of per-session state -
// regionsById (regionId -> {region: the wavesurfer Region instance, label:
// "healthy" | "defective" | "skip"}), overlaysByRegionId (regionId -> the
// spectrogram panel's mirrored overlay <div>), lastValidPositionByRegionId
// (the overlap guard's revert target), plus selection/playback/scroll-sync
// tracking. waveform.js and segments.js stay dumb renderers with no state of
// their own - a region's start/end lives on the wavesurfer Region object
// itself (the library's own source of truth), its label lives here.
//
// Mental model: the user draws free-form regions (no minimum duration -
// the backend already drops any region too short to yield a single
// chunk_audio window) and labels *intent* ("this stretch is healthy idle") -
// the fixed 1.0s/0.25s-step grid the model actually trains on is applied
// server-side at confirm time (see api/labeling.py's confirm_upload), never
// at labeling time.

import { uploadFile, fullSpectrogramUrl, confirmUpload } from "./api.js";
import {
  initWaveform,
  loadFileIntoWaveform,
  playRegion as playWaveformRegion,
  removeRegion as removeWaveformRegion,
  setRegionLabel,
  setRegionSelected,
  setRegionPosition as setWaveformRegionPosition,
  scrollToRegion,
  zoomTo,
  getDuration,
  setScroll as setWaveformScroll,
  play as playWaveform,
  pause as pauseWaveform,
  seekToStart as seekWaveformToStart,
  showSnapGuide,
  hideSnapGuide,
  LABEL_COLORS,
} from "./waveform.js";
import { renderRegionRows } from "./segments.js";

const STORAGE_PREFIX = "s1000_meta_";
const OVERLAP_TOAST_MESSAGE = "Regions can't overlap — reverted to previous position";
const OVERLAP_TOAST_DURATION_MS = 2500;
const SCROLL_SYNC_FALLBACK_MS = 100;
const SNAP_THRESHOLD_S = 0.1;

const tabLabelPanel = document.getElementById("tab-label");

const uploadForm = document.getElementById("upload-form");
const fileInput = document.getElementById("file-input");
const fileInputFilename = document.getElementById("file-input-filename");
const domainSelect = document.getElementById("domain-select");
const uploadButton = document.getElementById("upload-button");
const uploadStatus = document.getElementById("upload-status");
const metaClearButton = document.getElementById("meta-clear-button");

const NO_FILE_TEXT = "No file chosen";

// [{key: metadata field name sent to the server, elementId}] - text/number/
// textarea/select inputs whose .value maps straight through. The two
// "follow-up" fields (exhaust_custom, known_issues) are collected here too
// since they're plain text controls, but their value only ends up in the
// outgoing metadata conditionally - see collectRecordingMetadata().
const TEXT_LIKE_META_FIELDS = [
  { key: "bike_model", elementId: "meta-bike-model" },
  { key: "model_year", elementId: "meta-model-year" },
  { key: "kilometers_on_bike", elementId: "meta-kilometers-on-bike" },
  { key: "oil_type", elementId: "meta-oil-type" },
  { key: "kilometers_since_last_oilchange", elementId: "meta-kilometers-since-oilchange" },
  { key: "recording_device", elementId: "meta-recording-device" },
  { key: "contributor", elementId: "meta-contributor" },
  { key: "notes", elementId: "meta-notes" },
  { key: "exhaust_custom", elementId: "meta-exhaust-system" },
  { key: "known_issues", elementId: "meta-known-issues" },
];
const RADIO_META_GROUPS = [
  { key: "exhaust_is_stock", name: "meta-exhaust-stock" },
  { key: "known_issues_flag", name: "meta-known-issues-flag" },
];

const audioWorkspace = document.getElementById("audio-workspace");
const waveformContainer = document.getElementById("waveform");
const zoomSlider = document.getElementById("zoom-slider");

const transportRestartButton = document.getElementById("transport-restart-button");
const transportPlayButton = document.getElementById("transport-play-button");
const transportTime = document.getElementById("transport-time");

const overlapToast = document.getElementById("overlap-toast");

const spectrogramPanel = document.getElementById("spectrogram-panel"); // the scroll viewport itself
const spectrogramInner = document.getElementById("spectrogram-inner");
const spectrogramImage = document.getElementById("spectrogram-image");
const spectrogramRetryButton = document.getElementById("spectrogram-retry");

const regionsPanel = document.getElementById("regions-panel");
const regionsTbody = document.getElementById("regions-tbody");
const regionsEmptyHint = document.getElementById("regions-empty-hint");
const confirmButton = document.getElementById("confirm-button");
const confirmStatus = document.getElementById("confirm-status");

const sessionFilename = document.getElementById("session-filename");
const sessionDomain = document.getElementById("session-domain");
const sessionBike = document.getElementById("session-bike");

const confirmModal = document.getElementById("confirm-modal");
const confirmModalBreakdown = document.getElementById("confirm-modal-breakdown");
const confirmModalMetadata = document.getElementById("confirm-modal-metadata");
const confirmModalError = document.getElementById("confirm-modal-error");
const confirmModalCancel = document.getElementById("confirm-modal-cancel");
const confirmModalSave = document.getElementById("confirm-modal-save");

let currentUploadId = null;
let currentUploadMeta = null; // {filename, domain, sourceMetadata, segmentDuration, stepDuration}
let regionsById = new Map(); // regionId -> {region, label}
let overlaysByRegionId = new Map(); // regionId -> spectrogram overlay <div>
let lastValidPositionByRegionId = new Map(); // regionId -> {start, end}, the overlap guard's revert target
let warningRegionIds = new Set(); // regions a live drag elsewhere would currently overlap
let selectedRegionId = null;
let playingRegionId = null;
let isConfirmModalOpen = false;

let totalDurationSeconds = 0;
let isPlaybackActive = false;
let overlapToastTimeoutId = null;

// Guards the waveform<->spectrogram scroll mirroring against feedback loops:
// set before a programmatic scroll on either side, consumed by whichever
// handler sees it first (the echo, if the browser fires one) - a short
// fallback timeout clears it regardless, in case a given scroll happens to
// land on a position the browser doesn't consider a change (no echo fires).
let _scrollSyncing = false;
let scrollSyncTimeoutId = null;

export function initUploadTab() {
  restoreMetadataFromStorage();
  attachMetadataPersistence();
  updateFollowupVisibility();

  uploadForm.addEventListener("submit", handleUpload);
  fileInput.addEventListener("change", () => {
    fileInputFilename.textContent = fileInput.files[0]?.name || NO_FILE_TEXT;
  });
  metaClearButton.addEventListener("click", handleClearSavedMetadata);

  zoomSlider.addEventListener("input", () => zoomTo(Number(zoomSlider.value)));

  transportRestartButton.addEventListener("click", () => seekWaveformToStart());
  transportPlayButton.addEventListener("click", () => {
    if (isPlaybackActive) {
      pauseWaveform();
    } else {
      playWaveform();
    }
  });

  spectrogramImage.addEventListener("load", () => {
    spectrogramPanel.classList.add("is-loaded");
    spectrogramImage.classList.add("is-loaded");
  });
  spectrogramImage.addEventListener("error", () => {
    spectrogramPanel.classList.add("is-error");
  });
  spectrogramRetryButton.addEventListener("click", () => {
    if (currentUploadId) {
      loadFullSpectrogram(currentUploadId);
    }
  });
  spectrogramPanel.addEventListener("scroll", handleSpectrogramScroll);

  confirmButton.addEventListener("click", openConfirmModal);
  confirmModalCancel.addEventListener("click", closeConfirmModal);
  confirmModalSave.addEventListener("click", handleConfirmSave);
  confirmModal.addEventListener("click", (event) => {
    if (event.target === confirmModal) {
      closeConfirmModal();
    }
  });

  document.addEventListener("keydown", handleKeydown);
  window.addEventListener("beforeunload", handleBeforeUnload);
}

// ---------------------------------------------------------------------------
// Recording-details metadata: localStorage persistence
// ---------------------------------------------------------------------------

function safeGetItem(key) {
  try {
    return window.localStorage.getItem(STORAGE_PREFIX + key);
  } catch {
    return null;
  }
}

function safeSetItem(key, value) {
  try {
    window.localStorage.setItem(STORAGE_PREFIX + key, value);
  } catch {
    // localStorage unavailable (private browsing, storage full, ...) - the
    // form still works, it just won't remember values for next time.
  }
}

function safeRemoveItem(key) {
  try {
    window.localStorage.removeItem(STORAGE_PREFIX + key);
  } catch {
    // see safeSetItem
  }
}

function restoreMetadataFromStorage() {
  for (const field of TEXT_LIKE_META_FIELDS) {
    const stored = safeGetItem(field.key);
    if (stored !== null) {
      document.getElementById(field.elementId).value = stored;
    }
  }
  for (const group of RADIO_META_GROUPS) {
    const stored = safeGetItem(group.key);
    if (stored) {
      const radio = document.querySelector(`input[name="${group.name}"][value="${stored}"]`);
      if (radio) {
        radio.checked = true;
      }
    }
  }
}

function attachMetadataPersistence() {
  for (const field of TEXT_LIKE_META_FIELDS) {
    const element = document.getElementById(field.elementId);
    const eventName = element.tagName === "SELECT" ? "change" : "input";
    element.addEventListener(eventName, () => safeSetItem(field.key, element.value));
  }
  for (const group of RADIO_META_GROUPS) {
    for (const radio of document.querySelectorAll(`input[name="${group.name}"]`)) {
      radio.addEventListener("change", () => {
        safeSetItem(group.key, radio.value);
        updateFollowupVisibility();
      });
    }
  }
}

function handleClearSavedMetadata() {
  for (const field of TEXT_LIKE_META_FIELDS) {
    safeRemoveItem(field.key);
    const element = document.getElementById(field.elementId);
    if (element.tagName === "SELECT") {
      element.selectedIndex = 0;
    } else {
      element.value = "";
    }
  }
  for (const group of RADIO_META_GROUPS) {
    safeRemoveItem(group.key);
    for (const radio of document.querySelectorAll(`input[name="${group.name}"]`)) {
      radio.checked = false;
    }
  }
  updateFollowupVisibility();
}

function getRadioValue(name) {
  const checked = document.querySelector(`input[name="${name}"]:checked`);
  return checked ? checked.value : "";
}

// A follow-up field only ever holds a value while its condition is active -
// switching the radio away (in either direction, at any point) wipes both
// the DOM value and its localStorage entry immediately, so stale text from
// an abandoned choice can never resurface later, and never has to be
// filtered out at submit time either.
function updateFollowupVisibility() {
  const exhaustFollowupVisible = getRadioValue("meta-exhaust-stock") === "no";
  document.getElementById("meta-exhaust-followup").hidden = !exhaustFollowupVisible;
  if (!exhaustFollowupVisible) {
    clearFollowupField("meta-exhaust-system", "exhaust_custom");
  }

  const knownIssuesFollowupVisible = getRadioValue("meta-known-issues-flag") === "yes";
  document.getElementById("meta-known-issues-followup").hidden = !knownIssuesFollowupVisible;
  if (!knownIssuesFollowupVisible) {
    clearFollowupField("meta-known-issues", "known_issues");
  }
}

function clearFollowupField(elementId, storageKey) {
  document.getElementById(elementId).value = "";
  safeRemoveItem(storageKey);
}

function collectRecordingMetadata() {
  const metadata = {
    bike_model: document.getElementById("meta-bike-model").value.trim(),
    model_year: document.getElementById("meta-model-year").value.trim(),
    kilometers_on_bike: document.getElementById("meta-kilometers-on-bike").value.trim(),
    oil_type: document.getElementById("meta-oil-type").value.trim(),
    kilometers_since_last_oilchange: document.getElementById("meta-kilometers-since-oilchange").value.trim(),
    recording_device: document.getElementById("meta-recording-device").value.trim(),
    contributor: document.getElementById("meta-contributor").value.trim(),
    notes: document.getElementById("meta-notes").value.trim(),
  };

  metadata.exhaust_system =
    getRadioValue("meta-exhaust-stock") === "yes" ? "stock" : document.getElementById("meta-exhaust-system").value.trim();
  metadata.known_issues =
    getRadioValue("meta-known-issues-flag") === "yes" ? document.getElementById("meta-known-issues").value.trim() : "";

  return metadata;
}

// ---------------------------------------------------------------------------
// Upload
// ---------------------------------------------------------------------------

async function handleUpload(event) {
  event.preventDefault();

  const file = fileInput.files[0];
  if (!file) {
    uploadStatus.textContent = "Choose a file first.";
    return;
  }

  uploadButton.disabled = true;
  uploadStatus.textContent = "Uploading…";
  try {
    const response = await uploadFile(file, domainSelect.value || undefined, collectRecordingMetadata());

    currentUploadId = response.upload_id;
    currentUploadMeta = {
      filename: response.filename,
      domain: response.domain,
      sourceMetadata: response.source_metadata,
      segmentDuration: response.segment_duration_seconds,
      stepDuration: response.step_duration_seconds,
    };
    regionsById = new Map();
    overlaysByRegionId = new Map();
    lastValidPositionByRegionId = new Map();
    warningRegionIds = new Set();
    selectedRegionId = null;
    playingRegionId = null;
    isPlaybackActive = false;
    totalDurationSeconds = 0;
    spectrogramInner.querySelectorAll(".region-overlay").forEach((element) => element.remove());
    hideOverlapToast();

    uploadStatus.textContent = `Uploaded "${response.filename}" (${response.domain}).`;

    initWaveform(waveformContainer, {
      onRegionCreated: handleRegionCreated,
      onRegionChanged: handleRegionChanged,
      onRegionSettled: handleRegionSettled,
      onRegionRemoved: handleRegionRemoved,
      onRegionClicked: handleRegionClicked,
      onRegionDoubleClicked: handleRegionDoubleClicked,
      onBackgroundClicked: handleBackgroundClicked,
      onPlayStateChange: handlePlayStateChange,
      onPlaybackStateChange: handlePlaybackStateChange,
      onTimeUpdate: handleTimeUpdate,
      onScroll: handleWaveformScroll,
      onContentWidthChange: handleContentWidthChange,
    });
    zoomSlider.value = "0"; // 0 = fit the full file to the panel width
    updateTransportPlayButton(false);
    updateTransportTime(0);

    // Unhide before awaiting decode below: the content-width sync that
    // loadFileIntoWaveform triggers reads the waveform wrapper's clientWidth,
    // which is 0 for as long as an ancestor panel is still [hidden].
    audioWorkspace.hidden = false;
    regionsPanel.hidden = false;

    loadFullSpectrogram(currentUploadId);
    updateSessionHeader();
    refreshRegionsUI();

    await loadFileIntoWaveform(file);
    totalDurationSeconds = getDuration();
    updateTransportTime(0);
  } catch (error) {
    uploadStatus.textContent = `Upload failed: ${error.message}`;
  } finally {
    uploadButton.disabled = false;
  }
}

function loadFullSpectrogram(uploadId) {
  spectrogramPanel.classList.remove("is-loaded", "is-error");
  spectrogramImage.classList.remove("is-loaded");
  spectrogramImage.src = fullSpectrogramUrl(uploadId);
}

function updateSessionHeader() {
  sessionFilename.textContent = currentUploadMeta.filename;
  sessionDomain.textContent = currentUploadMeta.domain;
  const meta = currentUploadMeta.sourceMetadata || {};
  sessionBike.textContent = [meta.bike_model, meta.model_year].filter(Boolean).join(" · ");
}

// ---------------------------------------------------------------------------
// Global transport (plays/pauses the shared cursor position - separate from
// a region list row's own play button, which only plays that region's range)
// ---------------------------------------------------------------------------

function handlePlaybackStateChange(isPlaying) {
  isPlaybackActive = isPlaying;
  updateTransportPlayButton(isPlaying);
}

function updateTransportPlayButton(isPlaying) {
  transportPlayButton.textContent = isPlaying ? "■" : "▶";
  transportPlayButton.setAttribute("aria-label", isPlaying ? "Stop" : "Play");
}

function handleTimeUpdate(currentTime) {
  updateTransportTime(currentTime);
}

function updateTransportTime(currentTime) {
  transportTime.textContent = `${formatTime(currentTime)} / ${formatTime(totalDurationSeconds)}`;
}

function formatTime(seconds) {
  const total = Math.max(0, seconds || 0);
  const minutes = Math.floor(total / 60);
  const secs = total - minutes * 60;
  return `${String(minutes).padStart(2, "0")}:${secs.toFixed(1).padStart(4, "0")}`;
}

// ---------------------------------------------------------------------------
// Waveform <-> spectrogram bidirectional scroll sync
// ---------------------------------------------------------------------------

function beginScrollSync() {
  _scrollSyncing = true;
  clearTimeout(scrollSyncTimeoutId);
  scrollSyncTimeoutId = setTimeout(() => {
    _scrollSyncing = false;
  }, SCROLL_SYNC_FALLBACK_MS);
}

function handleWaveformScroll(scrollLeftPx) {
  if (_scrollSyncing) {
    _scrollSyncing = false;
    clearTimeout(scrollSyncTimeoutId);
    return;
  }
  beginScrollSync();
  spectrogramPanel.scrollLeft = scrollLeftPx;
}

function handleSpectrogramScroll() {
  if (_scrollSyncing) {
    _scrollSyncing = false;
    clearTimeout(scrollSyncTimeoutId);
    return;
  }
  beginScrollSync();
  setWaveformScroll(spectrogramPanel.scrollLeft);
}

// Keeps the spectrogram image (and, via percentage-based left/width, every
// region overlay drawn on it) the same pixel width as the waveform's current
// zoomed content width, so a shared scrollLeft always points at the same
// moment in time in both panels.
function handleContentWidthChange(widthPx) {
  if (widthPx > 0) {
    spectrogramInner.style.width = `${widthPx}px`;
  }
}

// ---------------------------------------------------------------------------
// Snap to adjacent region edges
// ---------------------------------------------------------------------------

// Every other region's start/end, plus the file boundaries - a settled (or
// in-progress) region's own start/end snaps to the nearest of these once
// within SNAP_THRESHOLD_S, letting regions sit back-to-back with no gap.
function collectSnapCandidates(excludeRegionId) {
  const candidates = [0, totalDurationSeconds];
  for (const [id, entry] of regionsById) {
    if (id === excludeRegionId) {
      continue;
    }
    candidates.push(entry.region.start, entry.region.end);
  }
  return candidates;
}

function findNearestSnapCandidate(time, candidates) {
  let nearest = null;
  let nearestDistance = SNAP_THRESHOLD_S;
  for (const candidate of candidates) {
    const distance = Math.abs(candidate - time);
    if (distance <= nearestDistance) {
      nearest = candidate;
      nearestDistance = distance;
    }
  }
  return nearest;
}

// Runs before the overlap check (so a snapped edge is what overlap gets
// judged against) - mutates the region in place via the wavesurfer API when
// either edge lands within threshold of a candidate.
function snapRegionToNeighbors(region) {
  const candidates = collectSnapCandidates(region.id);
  const snappedStart = findNearestSnapCandidate(region.start, candidates);
  const snappedEnd = findNearestSnapCandidate(region.end, candidates);
  if (snappedStart === null && snappedEnd === null) {
    return;
  }
  setWaveformRegionPosition(region, {
    start: snappedStart !== null ? snappedStart : region.start,
    end: snappedEnd !== null ? snappedEnd : region.end,
  });
}

// Live drag/resize feedback: a guide line at whichever edge (start,
// preferred, else end) is currently within snapping range of a candidate.
function updateSnapGuide(region) {
  const candidates = collectSnapCandidates(region.id);
  const nearStart = findNearestSnapCandidate(region.start, candidates);
  const nearEnd = nearStart === null ? findNearestSnapCandidate(region.end, candidates) : null;
  const guideTime = nearStart !== null ? nearStart : nearEnd;
  if (guideTime !== null) {
    showSnapGuide(guideTime);
  } else {
    hideSnapGuide();
  }
}

// ---------------------------------------------------------------------------
// Overlap protection
// ---------------------------------------------------------------------------

function findOverlappingRegionIds(excludeRegionId, start, end) {
  const overlapping = [];
  for (const [id, entry] of regionsById) {
    if (id === excludeRegionId) {
      continue;
    }
    const other = entry.region;
    if (start < other.end && end > other.start) {
      overlapping.push(id);
    }
  }
  return overlapping;
}

function showOverlapToast() {
  overlapToast.textContent = OVERLAP_TOAST_MESSAGE;
  overlapToast.classList.add("is-visible");
  clearTimeout(overlapToastTimeoutId);
  overlapToastTimeoutId = setTimeout(() => {
    overlapToast.classList.remove("is-visible");
  }, OVERLAP_TOAST_DURATION_MS);
}

function hideOverlapToast() {
  clearTimeout(overlapToastTimeoutId);
  overlapToast.classList.remove("is-visible");
}

// ---------------------------------------------------------------------------
// Region lifecycle (create / change / settle / remove / click / double-click)
// ---------------------------------------------------------------------------

function handleRegionCreated(region) {
  // A brand new region has no "previous position" to fall back to - if it
  // overlaps an existing one, it simply can't be created.
  const overlapping = findOverlappingRegionIds(region.id, region.start, region.end);
  if (overlapping.length > 0) {
    removeWaveformRegion(region);
    showOverlapToast();
    return;
  }
  lastValidPositionByRegionId.set(region.id, { start: region.start, end: region.end });
  regionsById.set(region.id, { region, label: "skip" });
  addSpectrogramOverlay(region, "skip");
  selectRegion(region.id);
  refreshRegionsUI();
}

// Live drag/resize tick (an existing region only - the plugin doesn't emit
// this during a brand-new region's own creation-drag): resync the overlay +
// table live, and flag any OTHER row the in-progress drag currently overlaps.
// The warning is cleared unconditionally once the drag settles, regardless
// of whether it was ultimately accepted or reverted - see handleRegionSettled.
function handleRegionChanged(region) {
  updateSpectrogramOverlayGeometry(region.id);
  warningRegionIds = new Set(findOverlappingRegionIds(region.id, region.start, region.end));
  renderRegionsTable();
  updateSnapGuide(region);
}

// Drag/resize settled: snaps to a nearby edge first (so overlap is judged
// against the snapped position), then the authoritative overlap check.
// Reverts to the last known non-overlapping position on conflict, otherwise
// commits the new position as the next revert target.
function handleRegionSettled(region) {
  warningRegionIds = new Set();
  hideSnapGuide();
  snapRegionToNeighbors(region);
  const overlapping = findOverlappingRegionIds(region.id, region.start, region.end);
  if (overlapping.length > 0) {
    const previous = lastValidPositionByRegionId.get(region.id);
    if (previous) {
      setWaveformRegionPosition(region, previous);
    }
    showOverlapToast();
  } else {
    lastValidPositionByRegionId.set(region.id, { start: region.start, end: region.end });
  }
  updateSpectrogramOverlayGeometry(region.id);
  renderRegionsTable();
}

function handleRegionRemoved(regionId) {
  regionsById.delete(regionId);
  removeSpectrogramOverlay(regionId);
  lastValidPositionByRegionId.delete(regionId);
  warningRegionIds.delete(regionId);
  if (selectedRegionId === regionId) {
    selectedRegionId = null;
  }
  if (playingRegionId === regionId) {
    playingRegionId = null;
  }
  refreshRegionsUI();
}

function handleRegionClicked(region) {
  selectRegion(region.id);
}

function handleRegionDoubleClicked(region) {
  playRegionById(region.id);
}

// Clicking empty waveform background deselects whatever was selected -
// wavesurfer's own click-to-seek still runs untouched alongside this.
function handleBackgroundClicked() {
  selectRegion(null);
}

function handlePlayStateChange(regionId) {
  playingRegionId = regionId;
  renderRegionsTable();
}

// ---------------------------------------------------------------------------
// Spectrogram region overlays
// ---------------------------------------------------------------------------

function addSpectrogramOverlay(region, label) {
  const overlay = document.createElement("div");
  overlay.className = "region-overlay";
  overlay.style.backgroundColor = LABEL_COLORS[label];
  // Selection only - creation stays waveform-only (the spectrogram has no
  // time-accurate drag-to-create mapping once zoomed).
  overlay.addEventListener("click", () => selectRegion(region.id));
  spectrogramInner.appendChild(overlay);
  overlaysByRegionId.set(region.id, overlay);
  updateOverlayGeometry(overlay, region);
}

function updateOverlayGeometry(overlay, region) {
  const duration = getDuration();
  if (!duration) {
    return;
  }
  overlay.style.left = `${(region.start / duration) * 100}%`;
  overlay.style.width = `${((region.end - region.start) / duration) * 100}%`;
}

function updateSpectrogramOverlayGeometry(regionId) {
  const overlay = overlaysByRegionId.get(regionId);
  const entry = regionsById.get(regionId);
  if (overlay && entry) {
    updateOverlayGeometry(overlay, entry.region);
  }
}

function updateSpectrogramOverlayLabel(regionId, label) {
  const overlay = overlaysByRegionId.get(regionId);
  if (overlay) {
    overlay.style.backgroundColor = LABEL_COLORS[label];
  }
}

function removeSpectrogramOverlay(regionId) {
  const overlay = overlaysByRegionId.get(regionId);
  if (overlay) {
    overlay.remove();
    overlaysByRegionId.delete(regionId);
  }
}

// ---------------------------------------------------------------------------
// Region list table + selection + actions
// ---------------------------------------------------------------------------

function sortedRegionViewModels() {
  return Array.from(regionsById.entries())
    .map(([id, entry]) => ({ id, start: entry.region.start, end: entry.region.end, label: entry.label }))
    .sort((a, b) => a.start - b.start);
}

function renderRegionsTable() {
  renderRegionRows(regionsTbody, sortedRegionViewModels(), {
    selectedRegionId,
    playingRegionId,
    warningRegionIds,
    onLabelChange: handleLabelChange,
    onPlay: playRegionById,
    onRemove: removeRegionById,
    onRowClick: handleRowClick,
  });
}

function refreshRegionsUI() {
  renderRegionsTable();
  updateConfirmAvailability();
  regionsEmptyHint.hidden = regionsById.size > 0;
}

function updateConfirmAvailability() {
  const hasLabeledRegion = Array.from(regionsById.values()).some((entry) => entry.label !== "skip");
  confirmButton.disabled = !hasLabeledRegion;
}

function handleLabelChange(regionId, label) {
  const entry = regionsById.get(regionId);
  if (!entry) {
    return;
  }
  entry.label = label;
  setRegionLabel(entry.region, label);
  updateSpectrogramOverlayLabel(regionId, label);
  renderRegionsTable();
  updateConfirmAvailability();
}

function selectRegion(regionId) {
  if (selectedRegionId) {
    const previous = regionsById.get(selectedRegionId);
    if (previous) {
      setRegionSelected(previous.region, false);
    }
  }
  selectedRegionId = regionId;
  if (regionId) {
    const next = regionsById.get(regionId);
    if (next) {
      setRegionSelected(next.region, true);
    }
  }
  renderRegionsTable();
}

function handleRowClick(regionId) {
  selectRegion(regionId);
  const entry = regionsById.get(regionId);
  if (entry) {
    scrollToRegion(entry.region);
  }
}

function playRegionById(regionId) {
  const entry = regionsById.get(regionId);
  if (entry) {
    playWaveformRegion(entry.region);
  }
}

function removeRegionById(regionId) {
  const entry = regionsById.get(regionId);
  if (entry) {
    removeWaveformRegion(entry.region); // triggers "region-removed" -> handleRegionRemoved
  }
}

// ---------------------------------------------------------------------------
// Confirm modal
// ---------------------------------------------------------------------------

// Mirrors chunk_audio's window count (src/audio_utils.py) in continuous time
// rather than samples - an estimate, same as the modal's own copy says, not
// a guarantee of the exact number api/labeling.py's confirm_upload will insert.
function estimateChunkCount(durationSeconds, segmentDuration, stepDuration) {
  if (durationSeconds < segmentDuration || stepDuration <= 0) {
    return 0;
  }
  return Math.floor((durationSeconds - segmentDuration) / stepDuration) + 1;
}

function pluralize(count, singularNoun) {
  return `${count} ${singularNoun}${count === 1 ? "" : "s"}`;
}

function openConfirmModal() {
  const labeledRegions = Array.from(regionsById.values()).filter((entry) => entry.label !== "skip");
  if (!currentUploadId || labeledRegions.length === 0) {
    return;
  }

  const totalRegionCount = regionsById.size;
  const skippedCount = totalRegionCount - labeledRegions.length;
  const estimatedChunks = labeledRegions.reduce(
    (sum, entry) =>
      sum + estimateChunkCount(entry.region.end - entry.region.start, currentUploadMeta.segmentDuration, currentUploadMeta.stepDuration),
    0
  );
  const countsByLabel = { healthy: 0, defective: 0 };
  for (const entry of labeledRegions) {
    countsByLabel[entry.label] += 1;
  }

  confirmModalBreakdown.innerHTML = "";

  const summaryLine = document.createElement("p");
  summaryLine.textContent =
    `${pluralize(totalRegionCount, "region")} → ~${pluralize(estimatedChunks, "segment")} ` +
    `(at ${currentUploadMeta.segmentDuration}s / ${currentUploadMeta.stepDuration}s step)`;
  confirmModalBreakdown.appendChild(summaryLine);

  const breakdownParts = [];
  if (countsByLabel.healthy > 0) {
    breakdownParts.push(pluralize(countsByLabel.healthy, "healthy region"));
  }
  if (countsByLabel.defective > 0) {
    breakdownParts.push(pluralize(countsByLabel.defective, "defective region"));
  }
  if (skippedCount > 0) {
    breakdownParts.push(pluralize(skippedCount, "skipped region"));
  }
  const breakdownLine = document.createElement("p");
  breakdownLine.textContent = breakdownParts.join(" · ");
  confirmModalBreakdown.appendChild(breakdownLine);

  const meta = currentUploadMeta.sourceMetadata || {};
  confirmModalMetadata.textContent = [currentUploadMeta.domain, meta.bike_model, meta.model_year].filter(Boolean).join(" · ");

  confirmModalError.hidden = true;
  confirmModalError.textContent = "";
  confirmModalSave.disabled = false;
  confirmModalSave.textContent = "Save to manifest";

  confirmModal.hidden = false;
  isConfirmModalOpen = true;
  confirmModalSave.focus();
}

function closeConfirmModal() {
  confirmModal.hidden = true;
  isConfirmModalOpen = false;
}

async function handleConfirmSave() {
  const regions = Array.from(regionsById.values())
    .filter((entry) => entry.label !== "skip")
    .map((entry) => ({ start_time: entry.region.start, end_time: entry.region.end, label: entry.label }));

  confirmModalSave.disabled = true;
  confirmModalSave.textContent = "Saving…";

  try {
    const response = await confirmUpload(currentUploadId, regions);
    const parts = [`Saved ${response.inserted} segment(s) from ${pluralize(regions.length, "region")}.`];
    if (response.skipped_regions > 0) {
      parts.push(`${pluralize(response.skipped_regions, "region")} too short - skipped.`);
    }
    if (response.failed.length) {
      parts.push(`${response.failed.length} failed - see console.`);
      console.error("Some chunks failed to confirm:", response.failed);
    }
    confirmStatus.textContent = parts.join(" ");
    closeConfirmModal();
    if (response.inserted > 0) {
      resetAfterConfirm();
    }
  } catch (error) {
    confirmModalError.textContent = `Confirm failed: ${error.message}`;
    confirmModalError.hidden = false;
    confirmModalSave.disabled = false;
    confirmModalSave.textContent = "Save to manifest";
  }
}

function resetAfterConfirm() {
  currentUploadId = null;
  currentUploadMeta = null;
  clearAllRegions();
  hideOverlapToast();
  audioWorkspace.hidden = true;
  regionsPanel.hidden = true;
  fileInput.value = "";
  fileInputFilename.textContent = NO_FILE_TEXT;
}

function clearAllRegions() {
  const entries = Array.from(regionsById.values());
  for (const entry of entries) {
    removeWaveformRegion(entry.region); // each remove synchronously runs handleRegionRemoved
  }
}

// ---------------------------------------------------------------------------
// beforeunload
// ---------------------------------------------------------------------------

function handleBeforeUnload(event) {
  if (regionsById.size > 0) {
    event.preventDefault();
    event.returnValue = "";
  }
}

// ---------------------------------------------------------------------------
// Keyboard shortcuts
// ---------------------------------------------------------------------------

function handleKeydown(event) {
  if (isConfirmModalOpen) {
    if (event.key === "Escape") {
      closeConfirmModal();
      event.preventDefault();
    }
    return;
  }

  if (tabLabelPanel.hidden || regionsPanel.hidden) {
    return;
  }

  const target = event.target;
  const isFormField = target instanceof HTMLElement && ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName);
  if (isFormField || event.ctrlKey || event.metaKey || event.altKey) {
    return;
  }

  const key = event.key.toLowerCase();

  switch (key) {
    case "h":
      labelSelectedRegion("healthy");
      event.preventDefault();
      break;
    case "d":
      labelSelectedRegion("defective");
      event.preventDefault();
      break;
    case "s":
      labelSelectedRegion("skip");
      event.preventDefault();
      break;
    case " ":
      toggleSelectedRegionPlayback();
      event.preventDefault();
      break;
    case "delete":
      removeSelectedRegion();
      event.preventDefault();
      break;
    case "tab":
      if (regionsById.size > 0) {
        selectAdjacentRegion(event.shiftKey ? -1 : 1);
        event.preventDefault();
      }
      break;
    default:
      break;
  }
}

function labelSelectedRegion(label) {
  if (!selectedRegionId) {
    return;
  }
  handleLabelChange(selectedRegionId, label);
}

function toggleSelectedRegionPlayback() {
  if (!selectedRegionId) {
    return;
  }
  playRegionById(selectedRegionId);
}

function removeSelectedRegion() {
  if (!selectedRegionId) {
    return;
  }
  removeRegionById(selectedRegionId);
}

function selectAdjacentRegion(direction) {
  const sortedIds = sortedRegionViewModels().map((viewModel) => viewModel.id);
  if (sortedIds.length === 0) {
    return;
  }
  const currentIndex = selectedRegionId ? sortedIds.indexOf(selectedRegionId) : -1;
  const nextIndex = Math.max(0, Math.min(sortedIds.length - 1, currentIndex + direction));
  const nextId = sortedIds[nextIndex];
  selectRegion(nextId);
  const entry = regionsById.get(nextId);
  if (entry) {
    scrollToRegion(entry.region);
  }
}
