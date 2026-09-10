// Segment grid: spectrogram image + play button + 3-way label toggle per
// segment. Deliberately dumb/stateless - takes a plain
// {segmentId, spectrogramUrl, currentLabel, startTime, endTime}[] plus two
// callbacks and owns no label state itself, so a future caller (e.g. Phase
// 3's crawler tab, feeding it rows that came from a download instead of a
// manual upload) can reuse it unchanged.

const LABELS = ["unlabeled", "healthy", "defective"];
const LABEL_TEXT = { unlabeled: "Skip", healthy: "Healthy", defective: "Defective" };

export function renderSegments(container, segments, { onLabelChange, onPlay } = {}) {
  container.innerHTML = "";
  for (const segment of segments) {
    container.appendChild(buildSegmentCard(segment, onLabelChange, onPlay));
  }
}

function buildSegmentCard(segment, onLabelChange, onPlay) {
  const card = document.createElement("div");
  card.className = "segment-card";
  card.dataset.segmentId = segment.segmentId;

  const img = document.createElement("img");
  img.src = segment.spectrogramUrl;
  img.alt = `Spectrogram, ${segment.startTime.toFixed(2)}s–${segment.endTime.toFixed(2)}s`;
  img.loading = "lazy";
  card.appendChild(img);

  const time = document.createElement("div");
  time.className = "segment-time";
  time.textContent = `${segment.startTime.toFixed(2)}s – ${segment.endTime.toFixed(2)}s`;
  card.appendChild(time);

  const actions = document.createElement("div");
  actions.className = "segment-actions";

  const playButton = document.createElement("button");
  playButton.type = "button";
  playButton.className = "play-button";
  playButton.textContent = "▶";
  playButton.addEventListener("click", () => onPlay?.(segment.segmentId, segment.startTime, segment.endTime));
  actions.appendChild(playButton);

  actions.appendChild(buildLabelToggle(segment, onLabelChange));
  card.appendChild(actions);

  return card;
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
    button.addEventListener("click", () => {
      toggle.querySelectorAll("button").forEach((b) => b.classList.remove("selected"));
      button.classList.add("selected");
      onLabelChange?.(segment.segmentId, label);
    });
    toggle.appendChild(button);
  }

  return toggle;
}
