// File input + domain select -> POST /uploads -> waveform + segment grid.
// Owns the label state (segmentId -> "healthy" | "defective"; unlabeled
// segments are simply absent from the map) since segments.js is a dumb
// renderer with no state of its own.

import { uploadFile, spectrogramUrl, confirmUpload } from "./api.js";
import { initWaveform, loadFileIntoWaveform, drawSegmentRegions, setRegionLabel, playSegment, playFromStart } from "./waveform.js";
import { renderSegments } from "./segments.js";

const fileInput = document.getElementById("file-input");
const domainSelect = document.getElementById("domain-select");
const uploadButton = document.getElementById("upload-button");
const uploadStatus = document.getElementById("upload-status");

const metadataFieldIds = {
  contributor: "meta-contributor",
  recording_device: "meta-recording-device",
  exhaust_system: "meta-exhaust-system",
  model_year: "meta-model-year",
  kilometers_on_bike: "meta-kilometers-on-bike",
  oil_type: "meta-oil-type",
  kilometers_since_last_oilchange: "meta-kilometers-since-oilchange",
  known_issues: "meta-known-issues",
  notes: "meta-notes",
};
const waveformPanel = document.getElementById("waveform-panel");
const waveformContainer = document.getElementById("waveform");
const playAllButton = document.getElementById("play-all-button");
const segmentsPanel = document.getElementById("segments-panel");
const segmentsGrid = document.getElementById("segments-grid");
const segmentsSummary = document.getElementById("segments-summary");
const confirmButton = document.getElementById("confirm-button");
const confirmStatus = document.getElementById("confirm-status");

let currentUploadId = null;
let currentSegments = []; // raw segments from the last UploadResponse
let labelsBySegmentId = new Map();

export function initUploadTab() {
  uploadButton.addEventListener("click", handleUpload);
  playAllButton.addEventListener("click", playFromStart);
  confirmButton.addEventListener("click", handleConfirm);
}

function collectRecordingMetadata() {
  const metadata = {};
  for (const [field, elementId] of Object.entries(metadataFieldIds)) {
    metadata[field] = document.getElementById(elementId).value.trim();
  }
  return metadata;
}

async function handleUpload() {
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
    labelsBySegmentId = new Map();

    uploadStatus.textContent =
      `Uploaded "${response.filename}" (${response.domain}) - ${response.segments.length} segments.`;

    initWaveform(waveformContainer);
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
    currentLabel: labelsBySegmentId.get(segment.segment_id) || "unlabeled",
    startTime: segment.start_time,
    endTime: segment.end_time,
  }));

  renderSegments(segmentsGrid, viewModels, {
    onLabelChange: handleLabelChange,
    onPlay: (_segmentId, startTime, endTime) => playSegment(startTime, endTime),
  });
  updateSummary();
}

function handleLabelChange(segmentId, label) {
  if (label === "unlabeled") {
    labelsBySegmentId.delete(segmentId);
  } else {
    labelsBySegmentId.set(segmentId, label);
  }
  setRegionLabel(segmentId, label);
  updateSummary();
}

function updateSummary() {
  const labeledCount = labelsBySegmentId.size;
  segmentsSummary.textContent = `${labeledCount} of ${currentSegments.length} labeled`;
  confirmButton.disabled = labeledCount === 0;
}

async function handleConfirm() {
  if (!currentUploadId || labelsBySegmentId.size === 0) {
    return;
  }

  const labels = Array.from(labelsBySegmentId, ([segment_id, label]) => ({ segment_id, label }));

  confirmButton.disabled = true;
  confirmStatus.textContent = "Saving…";
  try {
    const response = await confirmUpload(currentUploadId, labels);
    confirmStatus.textContent =
      `Saved ${response.inserted} segment(s).` +
      (response.failed.length ? ` ${response.failed.length} failed - see console.` : "");
    if (response.failed.length) {
      console.error("Some segments failed to confirm:", response.failed);
    }
    if (response.inserted > 0) {
      resetAfterConfirm();
    }
  } catch (error) {
    confirmStatus.textContent = `Confirm failed: ${error.message}`;
  } finally {
    confirmButton.disabled = labelsBySegmentId.size === 0;
  }
}

function resetAfterConfirm() {
  currentUploadId = null;
  currentSegments = [];
  labelsBySegmentId = new Map();
  segmentsGrid.innerHTML = "";
  segmentsPanel.hidden = true;
  waveformPanel.hidden = true;
  fileInput.value = "";
}
