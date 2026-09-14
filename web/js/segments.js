// Region list table: one row per user-drawn region, with a start/end/duration
// readout, a label dropdown, and play/remove actions. Presentation only -
// owns no state (upload.js's region map is the single source of truth) and
// is rebuilt wholesale on every change. That's deliberately simpler than the
// old per-segment card grid's surgical DOM updates (which existed only to
// avoid re-triggering lazy-loaded spectrogram <img> fetches) - a table of a
// few dozen plain text/dropdown rows is cheap to fully re-render, even on
// every pixel of a live region drag.

const LABELS = ["healthy", "defective", "skip"];
const LABEL_TEXT = { healthy: "Healthy", defective: "Defective", skip: "Skip" };
const PLAY_ICON = "▶"; // ▶
const STOP_ICON = "■"; // ■

// regions: [{id, start, end, label}], already sorted by start time - the "#"
// column and row order both follow that order directly. warningRegionIds
// (a Set, optional): regions that would be overlapped by a drag currently in
// progress elsewhere - highlighted here, cleared by the caller on drag end.
export function renderRegionRows(
  tbody,
  regions,
  { selectedRegionId, playingRegionId, warningRegionIds, onLabelChange, onPlay, onRemove, onRowClick } = {}
) {
  tbody.innerHTML = "";
  regions.forEach((region, index) => {
    tbody.appendChild(
      buildRow(region, index, { selectedRegionId, playingRegionId, warningRegionIds, onLabelChange, onPlay, onRemove, onRowClick })
    );
  });
}

function buildRow(region, index, { selectedRegionId, playingRegionId, warningRegionIds, onLabelChange, onPlay, onRemove, onRowClick }) {
  const row = document.createElement("tr");
  row.dataset.regionId = region.id;
  const classes = [];
  if (region.id === selectedRegionId) classes.push("is-selected");
  if (region.id === playingRegionId) classes.push("is-playing");
  if (warningRegionIds?.has(region.id)) classes.push("is-overlap-warning");
  row.className = classes.join(" ");

  row.appendChild(buildCell(String(index + 1)));
  row.appendChild(buildCell(`${region.start.toFixed(2)}s`));
  row.appendChild(buildCell(`${region.end.toFixed(2)}s`));
  row.appendChild(buildCell(`${(region.end - region.start).toFixed(2)}s`));
  row.appendChild(buildLabelCell(region, onLabelChange));
  row.appendChild(buildActionsCell(region, playingRegionId, onPlay, onRemove));

  row.addEventListener("click", () => onRowClick?.(region.id));

  return row;
}

function buildCell(text) {
  const td = document.createElement("td");
  td.textContent = text;
  return td;
}

function buildLabelCell(region, onLabelChange) {
  const td = document.createElement("td");
  const select = document.createElement("select");
  select.className = "region-label-select";
  select.dataset.label = region.label;
  for (const label of LABELS) {
    const option = document.createElement("option");
    option.value = label;
    option.textContent = LABEL_TEXT[label];
    option.selected = label === region.label;
    select.appendChild(option);
  }
  select.addEventListener("click", (event) => event.stopPropagation());
  select.addEventListener("change", () => {
    select.dataset.label = select.value;
    onLabelChange?.(region.id, select.value);
  });
  td.appendChild(select);
  return td;
}

function buildActionsCell(region, playingRegionId, onPlay, onRemove) {
  const td = document.createElement("td");
  td.className = "region-actions";

  const isPlaying = region.id === playingRegionId;
  const playButton = document.createElement("button");
  playButton.type = "button";
  playButton.className = "play-button";
  playButton.textContent = isPlaying ? STOP_ICON : PLAY_ICON;
  playButton.setAttribute("aria-label", isPlaying ? "Stop region" : "Play region");
  playButton.addEventListener("click", (event) => {
    event.stopPropagation();
    onPlay?.(region.id);
  });

  const removeButton = document.createElement("button");
  removeButton.type = "button";
  removeButton.className = "secondary-button remove-button";
  removeButton.textContent = "✕"; // ✕
  removeButton.setAttribute("aria-label", "Remove region");
  removeButton.addEventListener("click", (event) => {
    event.stopPropagation();
    onRemove?.(region.id);
  });

  td.append(playButton, removeButton);
  return td;
}
