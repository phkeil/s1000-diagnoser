// Diagnose tab: upload a recording, then show the full-file mel spectrogram
// with a time-aligned anomaly-score graph under it, a verdict summary, and a
// per-segment table. Read-only - nothing here can write to the manifest (the
// server keeps /diagnose/* on its own router and its own session store, see
// api/inference_tab.py).
//
// TIME ALIGNMENT is the whole point of the layout, and it is achieved purely
// with geometry rather than any shared coordinate library:
//
//   * Both the spectrogram and the graph live in their own scroll viewport,
//     and both viewports are the same width (they're siblings in one column).
//   * Both hold content of the SAME total width, laid out with the SAME left
//     and right gutters: PLOT_MARGIN_LEFT px of y-axis labels, then the plot
//     area, then PLOT_MARGIN_RIGHT px for the threshold label. The spectrogram
//     image sits in that middle band via padding on its wrapper; the SVG draws
//     its plot in exactly the same band.
//   * The SVG's viewBox is 1:1 with its CSS pixel width, so nothing is
//     stretched and a time t lands on the same x in both panels.
//   * scrollLeft is mirrored between the two viewports, so they stay aligned
//     once the content is wider than the panel (same guard-against-feedback
//     pattern the label tab uses - see web/js/upload.js).
//
// The content width is max(spectrogram natural width, panel width): 1px per
// mel hop (~11.6ms at hop_length=512 / 44.1kHz) for anything that fits, and a
// proportional stretch for a short file so it doesn't render as a sliver in a
// wide panel. Either way both panels get the identical number, so alignment
// holds at any width.

import { uploadForDiagnosis, getDiagnoseResults, diagnoseSpectrogramUrl } from "./api.js";

const SVG_NS = "http://www.w3.org/2000/svg";

const GRAPH_HEIGHT = 180;
const PLOT_MARGIN_LEFT = 46; // y-axis labels
const PLOT_MARGIN_RIGHT = 62; // "threshold" label, right of the dashed line
const PLOT_MARGIN_TOP = 12;
const PLOT_MARGIN_BOTTOM = 24; // x-axis labels

const Y_TICK_TARGET = 4;
const HEADROOM = 1.2; // y-axis tops out 20% above the tallest thing drawn
const MIN_X_TICK_SPACING_PX = 56;
const X_TICK_STEPS_S = [1, 2, 5, 10, 15, 30, 60, 120, 300];

const SCROLL_SYNC_FALLBACK_MS = 100;

const form = document.getElementById("diagnose-form");
const fileInput = document.getElementById("diagnose-file-input");
const fileName = document.getElementById("diagnose-file-name");
const domainSelect = document.getElementById("diagnose-domain-select");
const analyzeButton = document.getElementById("diagnose-button");
const statusText = document.getElementById("diagnose-status");

const resultsPanel = document.getElementById("diagnose-results");
const fileHeader = document.getElementById("diagnose-file-header");
const spectrogramScroll = document.getElementById("diagnose-spectrogram-scroll");
const spectrogramInner = document.getElementById("diagnose-spectrogram-inner");
const spectrogramImage = document.getElementById("diagnose-spectrogram-image");
const graphScroll = document.getElementById("diagnose-graph-scroll");
const graphSvg = document.getElementById("diagnose-graph");
const summaryBar = document.getElementById("diagnose-summary");
const tbody = document.getElementById("diagnose-tbody");

const NO_FILE_TEXT = "No file chosen";

let currentResults = null; // the last GET /diagnose/{id}/results body
let currentContentWidth = 0; // total px width shared by both panels

// Mirrors the label tab's guard: set before a programmatic scroll, consumed by
// whichever handler sees the echo first, with a timeout in case none fires.
let _scrollSyncing = false;
let scrollSyncTimeoutId = null;

export function initDiagnoseTab() {
  form.addEventListener("submit", handleAnalyze);
  fileInput.addEventListener("change", () => {
    fileName.textContent = fileInput.files[0]?.name || NO_FILE_TEXT;
  });

  spectrogramImage.addEventListener("load", () => {
    spectrogramScroll.classList.add("is-loaded");
    spectrogramImage.classList.add("is-loaded");
    relayout();
  });
  spectrogramImage.addEventListener("error", () => {
    spectrogramScroll.classList.add("is-error");
  });

  spectrogramScroll.addEventListener("scroll", () => syncScroll(spectrogramScroll, graphScroll));
  graphScroll.addEventListener("scroll", () => syncScroll(graphScroll, spectrogramScroll));

  window.addEventListener("resize", relayout);

  return { onTabShown: relayout };
}

// ---------------------------------------------------------------------------
// Upload -> spectrogram + results
// ---------------------------------------------------------------------------

async function handleAnalyze(event) {
  event.preventDefault();

  const file = fileInput.files[0];
  if (!file) {
    statusText.textContent = "Choose a file first.";
    return;
  }

  analyzeButton.disabled = true;
  statusText.textContent = "Uploading…";
  resetResultsPanel();

  try {
    const upload = await uploadForDiagnosis(file, domainSelect.value || undefined);

    renderFileHeader(upload);
    resultsPanel.hidden = false;
    statusText.textContent = "Analyzing…";
    setLoadingPlaceholder();

    // Both in flight at once: the spectrogram render and the model scoring
    // pass are independent server-side, and each is slow enough to be worth
    // not waiting on the other.
    loadSpectrogram(upload.upload_id);
    const results = await getDiagnoseResults(upload.upload_id);

    currentResults = results;
    relayout();
    renderSummary(results);
    renderTable(results);
    statusText.textContent = `Analyzed "${upload.filename}".`;
  } catch (error) {
    statusText.textContent = `Analysis failed: ${error.message}`;
    summaryBar.textContent = "";
    summaryBar.className = "diagnose-summary";
  } finally {
    analyzeButton.disabled = false;
  }
}

function resetResultsPanel() {
  resultsPanel.hidden = true;
  spectrogramScroll.scrollLeft = 0;
  graphScroll.scrollLeft = 0;
  currentResults = null;
  graphSvg.replaceChildren();
  tbody.replaceChildren();
  summaryBar.textContent = "";
  summaryBar.className = "diagnose-summary"
}

function setLoadingPlaceholder() {
  summaryBar.textContent = "Scoring segments…";
  summaryBar.className = "diagnose-summary is-pending";
}

function loadSpectrogram(uploadId) {
  spectrogramScroll.classList.remove("is-loaded", "is-error");
  spectrogramImage.classList.remove("is-loaded");
  spectrogramImage.src = diagnoseSpectrogramUrl(uploadId);
}

function renderFileHeader(upload) {
  fileHeader.replaceChildren(
    buildHeaderItem(`File: ${upload.filename}`),
    buildHeaderItem(`Domain: ${upload.domain}`),
    buildHeaderItem(`${upload.duration_seconds.toFixed(1)}s`)
  );
}

function buildHeaderItem(text) {
  const span = document.createElement("span");
  span.className = "session-item";
  span.textContent = text;
  return span;
}

// ---------------------------------------------------------------------------
// Layout: one content width, applied to both panels
// ---------------------------------------------------------------------------

function relayout() {
  const viewportWidth = spectrogramScroll.clientWidth;
  if (viewportWidth === 0) {
    return; // still inside a [hidden] ancestor - re-runs on tab show
  }

  const plotWidth = Math.max(
    spectrogramImage.naturalWidth || 0,
    viewportWidth - PLOT_MARGIN_LEFT - PLOT_MARGIN_RIGHT
  );
  currentContentWidth = PLOT_MARGIN_LEFT + plotWidth + PLOT_MARGIN_RIGHT;

  // box-sizing: border-box is global, so this width INCLUDES the padding -
  // the image itself (width: 100%) ends up exactly plotWidth wide, in the
  // same band the SVG plots into.
  spectrogramInner.style.width = `${currentContentWidth}px`;
  spectrogramInner.style.paddingLeft = `${PLOT_MARGIN_LEFT}px`;
  spectrogramInner.style.paddingRight = `${PLOT_MARGIN_RIGHT}px`;

  graphSvg.style.width = `${currentContentWidth}px`;
  graphSvg.setAttribute("width", String(currentContentWidth));
  graphSvg.setAttribute("height", String(GRAPH_HEIGHT));
  graphSvg.setAttribute("viewBox", `0 0 ${currentContentWidth} ${GRAPH_HEIGHT}`);

  if (currentResults) {
    renderGraph(currentResults, plotWidth);
  }
}

// ---------------------------------------------------------------------------
// Scroll sync
// ---------------------------------------------------------------------------

function syncScroll(source, target) {
  if (_scrollSyncing) {
    _scrollSyncing = false;
    clearTimeout(scrollSyncTimeoutId);
    return;
  }
  _scrollSyncing = true;
  clearTimeout(scrollSyncTimeoutId);
  scrollSyncTimeoutId = setTimeout(() => {
    _scrollSyncing = false;
  }, SCROLL_SYNC_FALLBACK_MS);
  target.scrollLeft = source.scrollLeft;
}

function scrollBothTo(scrollLeft) {
  const clamped = Math.max(0, Math.min(scrollLeft, currentContentWidth - spectrogramScroll.clientWidth));
  _scrollSyncing = false;
  clearTimeout(scrollSyncTimeoutId);
  spectrogramScroll.scrollLeft = clamped;
  graphScroll.scrollLeft = clamped;
}

// ---------------------------------------------------------------------------
// Anomaly score graph (hand-rolled SVG - the series is a few hundred points
// at most, so there is nothing here a charting library would do better)
// ---------------------------------------------------------------------------

function renderGraph(results, plotWidth) {
  const plotTop = PLOT_MARGIN_TOP;
  const plotBottom = GRAPH_HEIGHT - PLOT_MARGIN_BOTTOM;
  const plotHeight = plotBottom - plotTop;
  const duration = results.duration_seconds || 1;
  const segments = results.segments;

  // The threshold line is part of the chart, so it has to fit inside the
  // y-range even when every score sits far below it.
  const maxScore = segments.reduce((max, segment) => Math.max(max, segment.anomaly_score), 0);
  const yMax = Math.max(maxScore, results.threshold) * HEADROOM || 1;

  const timeToX = (time) => PLOT_MARGIN_LEFT + (time / duration) * plotWidth;
  const scoreToY = (score) => plotBottom - (Math.min(score, yMax) / yMax) * plotHeight;

  const children = [];

  // --- y grid + labels
  for (const tick of niceTicks(yMax, Y_TICK_TARGET)) {
    const y = scoreToY(tick);
    children.push(
      line(PLOT_MARGIN_LEFT, y, PLOT_MARGIN_LEFT + plotWidth, y, "diagnose-gridline"),
      svgText(PLOT_MARGIN_LEFT - 8, y + 4, formatTick(tick), "diagnose-axis-label diagnose-axis-label-y")
    );
  }

  // --- filled area, split into runs of same-verdict segments so each stretch
  // of the curve is colored by its own verdict, with no gap at a crossing
  // (each run reaches to the midpoint it shares with its neighbour).
  const points = segments.map((segment) => ({
    x: timeToX(segment.start_time),
    y: scoreToY(segment.anomaly_score),
    isAnomalous: segment.is_anomalous,
  }));

  for (const run of splitIntoVerdictRuns(points)) {
    children.push(
      polygon(
        [...run.points.map((point) => `${round(point.x)},${round(point.y)}`), `${round(run.endX)},${round(plotBottom)}`, `${round(run.startX)},${round(plotBottom)}`],
        run.isAnomalous ? "diagnose-area diagnose-area-anomaly" : "diagnose-area diagnose-area-healthy"
      )
    );
  }

  // --- score line
  if (points.length > 0) {
    const polyline = document.createElementNS(SVG_NS, "polyline");
    polyline.setAttribute("class", "diagnose-score-line");
    polyline.setAttribute("points", points.map((point) => `${round(point.x)},${round(point.y)}`).join(" "));
    children.push(polyline);
  }

  // --- threshold reference line
  const thresholdY = scoreToY(results.threshold);
  children.push(
    line(PLOT_MARGIN_LEFT, thresholdY, PLOT_MARGIN_LEFT + plotWidth, thresholdY, "diagnose-threshold-line"),
    svgText(PLOT_MARGIN_LEFT + plotWidth + 6, thresholdY + 4, "threshold", "diagnose-axis-label diagnose-threshold-label")
  );

  // --- x axis + ticks
  children.push(line(PLOT_MARGIN_LEFT, plotBottom, PLOT_MARGIN_LEFT + plotWidth, plotBottom, "diagnose-axis-line"));
  for (const tick of xTicks(duration, plotWidth)) {
    const x = timeToX(tick);
    children.push(
      line(x, plotBottom, x, plotBottom + 4, "diagnose-axis-line"),
      svgText(x, plotBottom + 16, `${formatTick(tick)}s`, "diagnose-axis-label diagnose-axis-label-x")
    );
  }

  // --- hover bands: one full-height, invisible column per segment carrying
  // the tooltip, so the whole column is a target rather than a 2px dot.
  points.forEach((point, index) => {
    const segment = segments[index];
    const left = index === 0 ? point.x : (points[index - 1].x + point.x) / 2;
    const right = index === points.length - 1 ? point.x : (point.x + points[index + 1].x) / 2;

    const dot = document.createElementNS(SVG_NS, "circle");
    dot.setAttribute("class", `diagnose-dot${segment.is_anomalous ? " is-anomaly" : ""}`);
    dot.setAttribute("cx", round(point.x));
    dot.setAttribute("cy", round(point.y));
    dot.setAttribute("r", "2");
    children.push(dot);

    const band = document.createElementNS(SVG_NS, "rect");
    band.setAttribute("class", "diagnose-hover-band");
    band.setAttribute("x", round(left));
    band.setAttribute("y", String(plotTop));
    band.setAttribute("width", round(Math.max(right - left, 1)));
    band.setAttribute("height", String(plotHeight));
    const title = document.createElementNS(SVG_NS, "title");
    title.textContent =
      `t=${segment.start_time.toFixed(2)}s  score=${segment.anomaly_score.toFixed(2)}  ` +
      `${segment.is_anomalous ? "anomaly" : "healthy"}`;
    band.appendChild(title);
    children.push(band);
  });

  graphSvg.replaceChildren(...children);
}

// [{points: [...], startX, endX, isAnomalous}] - consecutive points sharing a
// verdict, each run extended to the midpoint it shares with its neighbours so
// adjacent fills meet exactly instead of leaving a wedge of background.
function splitIntoVerdictRuns(points) {
  if (points.length === 0) {
    return [];
  }
  if (points.length === 1) {
    return [{ points, startX: points[0].x, endX: points[0].x, isAnomalous: points[0].isAnomalous }];
  }

  const runs = [];
  let current = { points: [points[0]], isAnomalous: points[0].isAnomalous };

  for (let index = 1; index < points.length; index += 1) {
    const point = points[index];
    if (point.isAnomalous === current.isAnomalous) {
      current.points.push(point);
      continue;
    }
    // The boundary point sits on both runs, at the midpoint between the two
    // samples, with its y linearly interpolated - so the fills share an edge.
    const previous = points[index - 1];
    const boundary = { x: (previous.x + point.x) / 2, y: (previous.y + point.y) / 2 };
    current.points.push(boundary);
    runs.push(current);
    current = { points: [boundary, point], isAnomalous: point.isAnomalous };
  }
  runs.push(current);

  return runs.map((run) => ({
    points: run.points,
    startX: run.points[0].x,
    endX: run.points[run.points.length - 1].x,
    isAnomalous: run.isAnomalous,
  }));
}

// ~targetCount ticks at a "nice" round step (1/2/2.5/5 x 10^n) covering
// [0, yMax]. The step is the candidate landing CLOSEST to targetCount rather
// than the next one up: rounding up unconditionally leaves as few as 3 labels
// for an awkward yMax (e.g. 12 -> a step of 5 -> 0/5/10).
function niceTicks(yMax, targetCount) {
  const magnitude = 10 ** Math.floor(Math.log10(yMax / targetCount));
  const niceStep = [1, 2, 2.5, 5, 10]
    .map((multiple) => multiple * magnitude)
    .reduce((best, step) => (Math.abs(yMax / step - targetCount) < Math.abs(yMax / best - targetCount) ? step : best));

  // Indexed rather than accumulated, so repeated += can't drift the labels
  // (0.1 + 0.1 + 0.1 = 0.30000000000000004).
  const ticks = [];
  for (let index = 0; index * niceStep <= yMax + niceStep * 1e-6; index += 1) {
    ticks.push(index * niceStep);
  }
  return ticks;
}

// 1s ticks by default, stepping up (2s, 5s, ...) as needed to keep labels from
// colliding on a long file.
function xTicks(duration, plotWidth) {
  const step =
    X_TICK_STEPS_S.find((candidate) => (candidate / duration) * plotWidth >= MIN_X_TICK_SPACING_PX) ??
    X_TICK_STEPS_S[X_TICK_STEPS_S.length - 1];

  const ticks = [];
  for (let time = 0; time <= duration; time += step) {
    ticks.push(time);
  }
  return ticks;
}

function formatTick(value) {
  return Number.isInteger(value) ? String(value) : value.toFixed(value < 1 ? 2 : 1);
}

function round(value) {
  return String(Math.round(value * 100) / 100);
}

function line(x1, y1, x2, y2, className) {
  const element = document.createElementNS(SVG_NS, "line");
  element.setAttribute("class", className);
  element.setAttribute("x1", round(x1));
  element.setAttribute("y1", round(y1));
  element.setAttribute("x2", round(x2));
  element.setAttribute("y2", round(y2));
  return element;
}

function svgText(x, y, content, className) {
  const element = document.createElementNS(SVG_NS, "text");
  element.setAttribute("class", className);
  element.setAttribute("x", round(x));
  element.setAttribute("y", round(y));
  element.textContent = content;
  return element;
}

function polygon(pointStrings, className) {
  const element = document.createElementNS(SVG_NS, "polygon");
  element.setAttribute("class", className);
  element.setAttribute("points", pointStrings.join(" "));
  return element;
}

// ---------------------------------------------------------------------------
// Summary bar + per-segment table
// ---------------------------------------------------------------------------

function formatPercent(value) {
  return `${Math.round(value * 100)}%`;
}

function renderSummary(results) {
  const anomalousCount = results.segments.filter((segment) => segment.is_anomalous).length;
  const isAnomaly = results.overall_verdict === "anomaly";

  summaryBar.className = `diagnose-summary ${isAnomaly ? "is-anomaly" : "is-healthy"}`;
  summaryBar.replaceChildren(
    buildSummaryItem(`Overall: ${isAnomaly ? "✗ ANOMALY DETECTED" : "✓ HEALTHY"}`, "diagnose-summary-verdict"),
    buildSummaryItem(`Confidence: ${formatPercent(results.overall_confidence)}`),
    buildSummaryItem(`${anomalousCount} / ${results.segments.length} segments anomalous`)
  );
}

function buildSummaryItem(content, className) {
  const span = document.createElement("span");
  span.className = className || "diagnose-summary-item";
  span.textContent = content;
  return span;
}

function renderTable(results) {
  tbody.replaceChildren(...results.segments.map((segment) => buildSegmentRow(segment)));
}

function buildSegmentRow(segment) {
  const row = document.createElement("tr");
  if (segment.is_anomalous) {
    row.className = "is-anomaly";
  }

  row.append(
    cell(String(segment.index + 1)),
    cell(`${segment.start_time.toFixed(2)}s`),
    cell(`${segment.end_time.toFixed(2)}s`),
    cell(segment.anomaly_score.toFixed(2)),
    verdictCell(segment.is_anomalous),
    cell(formatPercent(segment.confidence))
  );

  // Brings this segment's moment into view in both panels at once.
  row.tabIndex = 0;
  row.addEventListener("click", () => scrollToTime(segment.start_time));
  row.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      scrollToTime(segment.start_time);
    }
  });

  return row;
}

function cell(content) {
  const td = document.createElement("td");
  td.textContent = content;
  return td;
}

function verdictCell(isAnomalous) {
  const td = document.createElement("td");
  td.className = isAnomalous ? "diagnose-verdict-anomaly" : "diagnose-verdict-healthy";
  td.textContent = isAnomalous ? "✗ Anomaly" : "✓ Healthy";
  return td;
}

function scrollToTime(time) {
  if (!currentResults) {
    return;
  }
  const plotWidth = currentContentWidth - PLOT_MARGIN_LEFT - PLOT_MARGIN_RIGHT;
  const x = PLOT_MARGIN_LEFT + (time / (currentResults.duration_seconds || 1)) * plotWidth;
  scrollBothTo(x - spectrogramScroll.clientWidth / 2);
}
