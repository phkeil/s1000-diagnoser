// Segment grid: spectrogram image + play button + 3-way label toggle per
// segment. Presentation only - owns no *label* state (upload.js's
// labelsBySegmentId Map remains the single source of truth there), but does
// keep a segmentId -> card element lookup so upload.js can apply
// programmatic updates (bulk labeling, keyboard shortcuts, playback state)
// without rebuilding the DOM - a full re-render per label change would
// re-trigger every <img>'s network fetch, which is exactly what lazy-loading
// + the skeleton/retry states below are trying to avoid.

const LABELS = ["unlabeled", "healthy", "defective"];
const LABEL_TEXT = { unlabeled: "Skip", healthy: "Healthy", defective: "Defective" };
const PLAY_ICON = "▶"; // ▶
const STOP_ICON = "■"; // ■

let cardsBySegmentId = new Map();

export function renderSegments(container, segments, { onLabelChange, onPlay } = {}) {
  container.innerHTML = "";
  cardsBySegmentId = new Map();
  for (const segment of segments) {
    const card = buildSegmentCard(segment, onLabelChange, onPlay);
    cardsBySegmentId.set(segment.segmentId, card);
    container.appendChild(card);
  }
}

// Sets the toggle's selected label on one card without touching anything
// else (image, region, other cards) - the shared path for a manual click,
// a bulk action, and a keyboard shortcut alike.
export function setCardLabel(segmentId, label) {
  const card = cardsBySegmentId.get(segmentId);
  if (!card) {
    return;
  }
  card.querySelectorAll(".label-toggle button").forEach((button) => {
    button.classList.toggle("selected", button.dataset.label === label);
  });
}

// playingSegmentId: the one segment currently playing, or null if none.
// Clears the highlight/icon on every other card - only one segment plays at
// a time (see web/js/waveform.js).
export function setCardPlaying(playingSegmentId) {
  for (const [segmentId, card] of cardsBySegmentId) {
    const isPlaying = segmentId === playingSegmentId;
    card.classList.toggle("is-playing", isPlaying);
    const playButton = card.querySelector(".play-button");
    playButton.textContent = isPlaying ? STOP_ICON : PLAY_ICON;
    playButton.setAttribute("aria-label", isPlaying ? "Stop segment" : "Play segment");
  }
}

export function setCardVisible(segmentId, visible) {
  const card = cardsBySegmentId.get(segmentId);
  if (card) {
    card.hidden = !visible;
  }
}

export function getCardElement(segmentId) {
  return cardsBySegmentId.get(segmentId);
}

// DOM/render order, filtered to whatever isn't currently hidden by the
// active filter - the order arrow-key navigation walks.
export function getVisibleSegmentIdsInOrder() {
  return [...cardsBySegmentId.entries()].filter(([, card]) => !card.hidden).map(([segmentId]) => segmentId);
}

function buildSegmentCard(segment, onLabelChange, onPlay) {
  const card = document.createElement("div");
  card.className = "segment-card";
  card.dataset.segmentId = segment.segmentId;
  card.tabIndex = 0;

  card.appendChild(buildSpectrogramImage(segment));

  const time = document.createElement("div");
  time.className = "segment-time";
  time.textContent = `${segment.startTime.toFixed(2)}s – ${segment.endTime.toFixed(2)}s`;
  card.appendChild(time);

  const actions = document.createElement("div");
  actions.className = "segment-actions";

  const playButton = document.createElement("button");
  playButton.type = "button";
  playButton.className = "play-button";
  playButton.textContent = PLAY_ICON;
  playButton.setAttribute("aria-label", "Play segment");
  playButton.addEventListener("click", () => onPlay?.(segment.segmentId, segment.startTime, segment.endTime));
  actions.appendChild(playButton);

  actions.appendChild(buildLabelToggle(segment, onLabelChange));
  card.appendChild(actions);

  return card;
}

// While loading: skeleton shimmer (CSS, driven by the absence of is-loaded/
// is-error). On error: a retry button that re-triggers the fetch via a
// cache-busting query param, instead of a permanent broken-image icon.
function buildSpectrogramImage(segment) {
  const wrap = document.createElement("div");
  wrap.className = "spectrogram-wrap";

  const img = document.createElement("img");
  img.loading = "lazy";
  img.alt = `Spectrogram, ${segment.startTime.toFixed(2)}s–${segment.endTime.toFixed(2)}s`;
  img.addEventListener("load", () => wrap.classList.add("is-loaded"));
  img.addEventListener("error", () => wrap.classList.add("is-error"));
  img.src = segment.spectrogramUrl;

  const retryButton = document.createElement("button");
  retryButton.type = "button";
  retryButton.className = "spectrogram-retry";
  retryButton.textContent = "↺ Retry"; // ↺ Retry
  retryButton.addEventListener("click", () => {
    wrap.classList.remove("is-error", "is-loaded");
    const separator = segment.spectrogramUrl.includes("?") ? "&" : "?";
    img.src = `${segment.spectrogramUrl}${separator}retry=${Date.now()}`;
  });

  wrap.append(img, retryButton);
  return wrap;
}

function buildLabelToggle(segment, onLabelChange) {
  const toggle = document.createElement("div");
  toggle.className = "label-toggle";

  for (const label of LABELS) {
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.label = label;
    button.textContent = LABEL_TEXT[label];
    if (label === segment.currentLabel) {
      button.classList.add("selected");
    }
    button.addEventListener("click", () => onLabelChange?.(segment.segmentId, label));
    toggle.appendChild(button);
  }

  return toggle;
}
