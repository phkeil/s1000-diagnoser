// Upload form + segment grid: owns every piece of per-session state -
// currentSegments (the last UploadResponse), labelsBySegmentId (segmentId ->
// "healthy" | "defective"; an unlabeled segment is simply absent),
// skippedSegmentIds (segmentId explicitly set to "Skip" - tracked apart from
// "never touched" purely so the progress breakdown can tell the two apart),
// plus the active filter and keyboard focus/hover tracking. segments.js and
// waveform.js stay dumb renderers with no state of their own.

import { uploadFile, spectrogramUrl, confirmUpload } from "./api.js";
import {
  initWaveform,
  loadFileIntoWaveform,
  drawSegmentRegions,
  setRegionLabel,
  playSegment,
  playFromStart,
} from "./waveform.js";
import {
  renderSegments,
  setCardLabel,
  setCardPlaying,
  setCardVisible,
  getCardElement,
  getVisibleSegmentIdsInOrder,
} from "./segments.js";

const STORAGE_PREFIX = "s1000_meta_";

const tabLabelPanel = document.getElementById("tab-label");

const uploadForm = document.getElementById("upload-form");
const fileInput = document.getElementById("file-input");
const fileInputFilename = document.getElementById("file-input-filename");
const domainSelect = document.getElementById("domain-select");
const uploadButton = document.getElementById("upload-button");
const uploadStatus = document.getElementById("upload-status");
const metaClearButton = document.getElementById("meta-clear-button");

const NO_FILE_TEXT = "Keine Datei ausgewählt";

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

const waveformPanel = document.getElementById("waveform-panel");
const waveformContainer = document.getElementById("waveform");
const playAllButton = document.getElementById("play-all-button");

const segmentsPanel = document.getElementById("segments-panel");
const segmentsGrid = document.getElementById("segments-grid");

const bulkButtons = document.querySelectorAll(".bulk-button");
const filterButtons = document.querySelectorAll(".filter-button");
const bulkProgress = document.getElementById("bulk-progress");
const confirmButton = document.getElementById("confirm-button");
const confirmStatus = document.getElementById("confirm-status");

const sessionFilename = document.getElementById("session-filename");
const sessionDomain = document.getElementById("session-domain");
const sessionBike = document.getElementById("session-bike");
const sessionProgress = document.getElementById("session-progress");

const confirmModal = document.getElementById("confirm-modal");
const confirmModalBreakdown = document.getElementById("confirm-modal-breakdown");
const confirmModalMetadata = document.getElementById("confirm-modal-metadata");
const confirmModalError = document.getElementById("confirm-modal-error");
const confirmModalCancel = document.getElementById("confirm-modal-cancel");
const confirmModalSave = document.getElementById("confirm-modal-save");

let currentUploadId = null;
let currentSegments = []; // raw segments from the last UploadResponse
let currentUploadMeta = null; // {filename, domain, sourceMetadata}
let labelsBySegmentId = new Map();
let skippedSegmentIds = new Set();
let currentFilter = "all";
let isConfirmModalOpen = false;

let hoveredSegmentId = null;
let focusedSegmentId = null;

export function initUploadTab() {
  restoreMetadataFromStorage();
  attachMetadataPersistence();
  updateFollowupVisibility();

  uploadForm.addEventListener("submit", handleUpload);
  fileInput.addEventListener("change", () => {
    fileInputFilename.textContent = fileInput.files[0]?.name || NO_FILE_TEXT;
  });
  metaClearButton.addEventListener("click", handleClearSavedMetadata);
  playAllButton.addEventListener("click", playFromStart);

  for (const button of bulkButtons) {
    button.addEventListener("click", () => {
      const label = button.dataset.label;
      if (button.dataset.bulkScope === "all") {
        bulkLabelAll(label);
      } else {
        bulkLabelUnlabeledOnly(label);
      }
    });
  }
  for (const button of filterButtons) {
    button.addEventListener("click", () => applyFilter(button.dataset.filter));
  }

  confirmButton.addEventListener("click", openConfirmModal);
  confirmModalCancel.addEventListener("click", closeConfirmModal);
  confirmModalSave.addEventListener("click", handleConfirmSave);
  confirmModal.addEventListener("click", (event) => {
    if (event.target === confirmModal) {
      closeConfirmModal();
    }
  });

  segmentsGrid.addEventListener("mouseover", (event) => {
    const card = event.target.closest(".segment-card");
    if (card) {
      hoveredSegmentId = card.dataset.segmentId;
    }
  });
  segmentsGrid.addEventListener("mouseleave", () => {
    hoveredSegmentId = null;
  });
  segmentsGrid.addEventListener("focusin", (event) => {
    const card = event.target.closest(".segment-card");
    if (card) {
      focusedSegmentId = card.dataset.segmentId;
    }
  });
  segmentsGrid.addEventListener("focusout", (event) => {
    if (!segmentsGrid.contains(event.relatedTarget)) {
      focusedSegmentId = null;
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
  const exhaustFollowupVisible = getRadioValue("meta-exhaust-stock") === "nein";
  document.getElementById("meta-exhaust-followup").hidden = !exhaustFollowupVisible;
  if (!exhaustFollowupVisible) {
    clearFollowupField("meta-exhaust-system", "exhaust_custom");
  }

  const knownIssuesFollowupVisible = getRadioValue("meta-known-issues-flag") === "ja";
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
    getRadioValue("meta-exhaust-stock") === "ja" ? "stock" : document.getElementById("meta-exhaust-system").value.trim();
  metadata.known_issues =
    getRadioValue("meta-known-issues-flag") === "ja" ? document.getElementById("meta-known-issues").value.trim() : "";

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
    currentSegments = response.segments;
    currentUploadMeta = { filename: response.filename, domain: response.domain, sourceMetadata: response.source_metadata };
    labelsBySegmentId = new Map();
    skippedSegmentIds = new Set();

    uploadStatus.textContent =
      `Uploaded "${response.filename}" (${response.domain}) - ${response.segments.length} segments.`;

    initWaveform(waveformContainer, { onPlayStateChange: setCardPlaying });
    await loadFileIntoWaveform(file);
    waveformPanel.hidden = false;

    drawSegmentRegions(
      currentSegments.map((segment) => ({
        segmentId: segment.segment_id,
        startTime: segment.start_time,
        endTime: segment.end_time,
        label: "unlabeled",
      }))
    );

    renderGrid();
    applyFilter("all");
    updateSessionHeader();
    updateProgress();
    segmentsPanel.hidden = false;
  } catch (error) {
    uploadStatus.textContent = `Upload failed: ${error.message}`;
  } finally {
    uploadButton.disabled = false;
  }
}

function renderGrid() {
  const viewModels = currentSegments.map((segment) => ({
    segmentId: segment.segment_id,
    spectrogramUrl: spectrogramUrl(segment.segment_id),
    currentLabel: "unlabeled",
    startTime: segment.start_time,
    endTime: segment.end_time,
  }));

  renderSegments(segmentsGrid, viewModels, {
    onLabelChange: handleLabelChange,
    onPlay: (segmentId, startTime, endTime) => playSegment(segmentId, startTime, endTime),
  });
}

function updateSessionHeader() {
  sessionFilename.textContent = currentUploadMeta.filename;
  sessionDomain.textContent = currentUploadMeta.domain;
  const meta = currentUploadMeta.sourceMetadata || {};
  sessionBike.textContent = [meta.bike_model, meta.model_year].filter(Boolean).join(" · ");
}

// ---------------------------------------------------------------------------
// Labeling (single-card, bulk, and keyboard shortcuts all funnel through
// setLabel so the card visuals, waveform region, and label state can never
// drift apart)
// ---------------------------------------------------------------------------

function setLabel(segmentId, label) {
  if (label === "unlabeled") {
    labelsBySegmentId.delete(segmentId);
    skippedSegmentIds.add(segmentId);
  } else {
    labelsBySegmentId.set(segmentId, label);
    skippedSegmentIds.delete(segmentId);
  }
  setCardLabel(segmentId, label);
  setRegionLabel(segmentId, label);
}

function afterLabelsChanged() {
  updateProgress();
  applyFilter(currentFilter);
}

function handleLabelChange(segmentId, label) {
  setLabel(segmentId, label);
  afterLabelsChanged();
}

function bulkLabelAll(label) {
  for (const segment of currentSegments) {
    setLabel(segment.segment_id, label);
  }
  afterLabelsChanged();
}

function bulkLabelUnlabeledOnly(label) {
  for (const segment of currentSegments) {
    if (!labelsBySegmentId.has(segment.segment_id)) {
      setLabel(segment.segment_id, label);
    }
  }
  afterLabelsChanged();
}

function applyFilter(filterValue) {
  currentFilter = filterValue;
  for (const button of filterButtons) {
    button.classList.toggle("active", button.dataset.filter === filterValue);
  }
  for (const segment of currentSegments) {
    const label = labelsBySegmentId.get(segment.segment_id) || "unlabeled";
    setCardVisible(segment.segment_id, filterValue === "all" || label === filterValue);
  }
}

function updateProgress() {
  const total = currentSegments.length;
  let healthy = 0;
  let defective = 0;
  for (const label of labelsBySegmentId.values()) {
    if (label === "healthy") {
      healthy += 1;
    } else if (label === "defective") {
      defective += 1;
    }
  }
  const skipped = skippedSegmentIds.size;
  const reviewed = healthy + defective + skipped;
  const unlabeled = total - reviewed;

  const summary = `${reviewed} of ${total} reviewed — ${healthy} healthy · ${defective} defective · ${skipped} skipped · ${unlabeled} unlabeled`;
  bulkProgress.textContent = summary;
  sessionProgress.textContent = summary;
  confirmButton.disabled = labelsBySegmentId.size === 0;

  return { total, healthy, defective, skipped, unlabeled };
}

// ---------------------------------------------------------------------------
// Confirm modal
// ---------------------------------------------------------------------------

function openConfirmModal() {
  if (!currentUploadId || labelsBySegmentId.size === 0) {
    return;
  }

  const { total, healthy, defective, skipped, unlabeled } = updateProgress();
  confirmModalBreakdown.innerHTML = "";
  const rows = [
    ["Healthy", healthy],
    ["Defective", defective],
    ["Skipped", skipped],
    ["Unlabeled (will be skipped)", unlabeled],
    ["Total segments", total],
  ];
  for (const [label, value] of rows) {
    const dt = document.createElement("dt");
    dt.textContent = label;
    const dd = document.createElement("dd");
    dd.textContent = String(value);
    confirmModalBreakdown.append(dt, dd);
  }

  const meta = currentUploadMeta.sourceMetadata || {};
  confirmModalMetadata.textContent = [currentUploadMeta.domain, meta.bike_model, meta.model_year]
    .filter(Boolean)
    .join(" · ");

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
  const labels = Array.from(labelsBySegmentId, ([segment_id, label]) => ({ segment_id, label }));

  confirmModalSave.disabled = true;
  confirmModalSave.textContent = "Saving…";

  try {
    const response = await confirmUpload(currentUploadId, labels);
    confirmStatus.textContent =
      `Saved ${response.inserted} segment(s).` +
      (response.failed.length ? ` ${response.failed.length} failed - see console.` : "");
    if (response.failed.length) {
      console.error("Some segments failed to confirm:", response.failed);
    }
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
  currentSegments = [];
  currentUploadMeta = null;
  labelsBySegmentId = new Map();
  skippedSegmentIds = new Set();
  segmentsGrid.innerHTML = "";
  segmentsPanel.hidden = true;
  waveformPanel.hidden = true;
  fileInput.value = "";
  fileInputFilename.textContent = NO_FILE_TEXT;
}

// ---------------------------------------------------------------------------
// beforeunload
// ---------------------------------------------------------------------------

function handleBeforeUnload(event) {
  const hasUnsavedWork = currentSegments.length > 0 && (labelsBySegmentId.size > 0 || skippedSegmentIds.size > 0);
  if (hasUnsavedWork) {
    event.preventDefault();
    event.returnValue = "";
  }
}

// ---------------------------------------------------------------------------
// Keyboard shortcuts
// ---------------------------------------------------------------------------

function getActiveSegmentId() {
  return focusedSegmentId || hoveredSegmentId;
}

function handleKeydown(event) {
  if (isConfirmModalOpen) {
    if (event.key === "Escape") {
      closeConfirmModal();
      event.preventDefault();
    }
    return;
  }

  if (tabLabelPanel.hidden || segmentsPanel.hidden) {
    return;
  }

  const target = event.target;
  const isFormField = target instanceof HTMLElement && ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName);
  if (isFormField || event.ctrlKey || event.metaKey || event.altKey) {
    return;
  }

  const key = event.key.toLowerCase();

  if (event.shiftKey && key === "h") {
    bulkLabelUnlabeledOnly("healthy");
    event.preventDefault();
    return;
  }
  if (event.shiftKey && key === "d") {
    bulkLabelUnlabeledOnly("defective");
    event.preventDefault();
    return;
  }

  switch (key) {
    case "h":
      labelActiveSegment("healthy");
      event.preventDefault();
      break;
    case "d":
      labelActiveSegment("defective");
      event.preventDefault();
      break;
    case "s":
      labelActiveSegment("unlabeled");
      event.preventDefault();
      break;
    case " ":
      togglePlayActiveSegment();
      event.preventDefault();
      break;
    case "arrowright":
      focusAdjacentCard(1);
      event.preventDefault();
      break;
    case "arrowleft":
      focusAdjacentCard(-1);
      event.preventDefault();
      break;
    default:
      break;
  }
}

function labelActiveSegment(label) {
  const segmentId = getActiveSegmentId();
  if (!segmentId) {
    return;
  }
  handleLabelChange(segmentId, label);
}

function togglePlayActiveSegment() {
  const segmentId = getActiveSegmentId();
  if (!segmentId) {
    return;
  }
  const segment = currentSegments.find((s) => s.segment_id === segmentId);
  if (!segment) {
    return;
  }
  playSegment(segmentId, segment.start_time, segment.end_time);
}

function focusAdjacentCard(direction) {
  const visibleIds = getVisibleSegmentIdsInOrder();
  if (visibleIds.length === 0) {
    return;
  }
  const activeId = getActiveSegmentId();
  const currentIndex = activeId ? visibleIds.indexOf(activeId) : -1;
  const nextIndex = Math.max(0, Math.min(visibleIds.length - 1, currentIndex + direction));
  getCardElement(visibleIds[nextIndex])?.focus();
}
