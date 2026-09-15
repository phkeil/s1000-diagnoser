// DOM: loading state, upload form, plain-language verdict, spectrogram +
// time-aligned score graph, per-segment table.
//
// The page is written for a rider, not for a developer: the verdict leads, the
// spectrogram and the per-segment numbers sit behind "what am I looking at?"
// and "show details" toggles, and nothing on screen names a library.
//
// The graph and the time alignment are a port of web/js/diagnose.js, and the
// geometry contract is the same one documented there:
//
//   * The spectrogram and the graph each live in their own scroll viewport,
//     both the same width, both holding content of the SAME total width with
//     the SAME left/right gutters: PLOT_MARGIN_LEFT px of y-axis labels, the
//     plot area, then PLOT_MARGIN_RIGHT px for the threshold label.
//   * The SVG viewBox is 1:1 with its CSS pixel width, so a time t lands on
//     the same x in both panels.
//   * scrollLeft is mirrored between the viewports, with the same
//     guard-against-feedback flag the tool app uses.
//
// The content width is max(spectrogram natural width, panel width): 1px per
// mel hop for anything that fits, and a proportional stretch for a short file
// so it doesn't render as a sliver.

const SVG_NS = "http://www.w3.org/2000/svg";

const GRAPH_HEIGHT = 180;
const PLOT_MARGIN_LEFT = 46;
const PLOT_MARGIN_RIGHT = 62;
const PLOT_MARGIN_TOP = 12;
const PLOT_MARGIN_BOTTOM = 24;

const Y_TICK_TARGET = 4;
const HEADROOM = 1.2;
const MIN_X_TICK_SPACING_PX = 56;
const X_TICK_STEPS_S = [1, 2, 5, 10, 15, 30, 60, 120, 300];

const SCROLL_SYNC_FALLBACK_MS = 100;

const NO_FILE_TEXT = "No file chosen";

const elements = {};
let currentResults = null;
let currentContentWidth = 0;
let spectrogramUrl = null;

let _scrollSyncing = false;
let scrollSyncTimeoutId = null;

export function init({ onAnalyze }) {
  elements.bootPanel = document.getElementById("boot-panel");
  elements.bootMessage = document.getElementById("boot-message");
  elements.bootDetail = document.getElementById("boot-detail");
  elements.bootBar = document.getElementById("boot-bar");
  elements.bootBarFill = document.getElementById("boot-bar-fill");
  elements.form = document.getElementById("analyze-form");
  elements.fileInput = document.getElementById("file-input");
  elements.fileName = document.getElementById("file-name");
  elements.dropZone = document.getElementById("drop-zone");
  elements.domainRadios = Array.from(document.querySelectorAll('input[name="domain"]'));
  elements.button = document.getElementById("analyze-button");
  elements.status = document.getElementById("analysis-status");
  elements.results = document.getElementById("results");
  elements.verdict = document.getElementById("verdict");
  elements.fileHeader = document.getElementById("file-header");
  elements.spectrogramScroll = document.getElementById("spectrogram-scroll");
  elements.spectrogramInner = document.getElementById("spectrogram-inner");
  elements.spectrogramImage = document.getElementById("spectrogram-image");
  elements.graphScroll = document.getElementById("graph-scroll");
  elements.graph = document.getElementById("graph");
  elements.summary = document.getElementById("summary");
  elements.tbody = document.getElementById("segment-tbody");

  elements.form.addEventListener("submit", (event) => {
    event.preventDefault();
    const file = elements.fileInput.files[0];
    if (!file) {
      setAnalysisStatus("Choose a file first.", true);
      return;
    }
    onAnalyze(file, selectedDomain());
  });

  elements.fileInput.addEventListener("change", () => {
    showChosenFile(elements.fileInput.files[0]);
  });

  elements.dropZone.addEventListener("click", () => elements.fileInput.click());
  elements.dropZone.addEventListener("dragover", (event) => {
    event.preventDefault();
    elements.dropZone.classList.add("is-dragover");
  });
  elements.dropZone.addEventListener("dragleave", () => elements.dropZone.classList.remove("is-dragover"));
  elements.dropZone.addEventListener("drop", (event) => {
    event.preventDefault();
    elements.dropZone.classList.remove("is-dragover");
    const file = event.dataTransfer.files[0];
    if (file) {
      elements.fileInput.files = event.dataTransfer.files;
      showChosenFile(file);
    }
  });

  elements.spectrogramImage.addEventListener("load", relayout);
  elements.spectrogramScroll.addEventListener("scroll", () =>
    syncScroll(elements.spectrogramScroll, elements.graphScroll)
  );
  elements.graphScroll.addEventListener("scroll", () =>
    syncScroll(elements.graphScroll, elements.spectrogramScroll)
  );
  window.addEventListener("resize", relayout);
}

function showChosenFile(file) {
  elements.fileName.textContent = file ? file.name : NO_FILE_TEXT;
  elements.fileName.classList.toggle("has-file", Boolean(file));
}

function selectedDomain() {
  return elements.domainRadios.find((radio) => radio.checked)?.value;
}

// ---------------------------------------------------------------------------
// Status
// ---------------------------------------------------------------------------

// Boot is slow enough (a 110 MB model plus a WASM Python runtime) that a bare
// spinner reads as a hang, so the page shows how far along it is. `fraction` is
// null while there is nothing to measure, which the bar renders as a sweep.
export function setBootProgress({ fraction, message, detail }) {
  elements.bootMessage.textContent = message;
  elements.bootDetail.textContent = detail || "";

  const isIndeterminate = fraction === null || fraction === undefined;
  elements.bootBar.classList.toggle("is-indeterminate", isIndeterminate);

  if (isIndeterminate) {
    elements.bootBar.removeAttribute("aria-valuenow");
    return;
  }

  const percent = Math.max(0, Math.min(100, Math.round(fraction * 100)));
  elements.bootBarFill.style.width = percent + "%";
  elements.bootBar.setAttribute("aria-valuenow", String(percent));
}

// The analyzer takes ten seconds or so to come up. Until it does there is a
// progress bar and nothing to interact with; the upload form only appears once
// both workers can actually serve a file.
export function setReady() {
  elements.bootPanel.hidden = true;
  elements.form.hidden = false;
  elements.button.disabled = false;
  setAnalysisStatus("Your recording is analyzed on your device via WebGPU in this browser tab. Nothing is uploaded.");
}

export function setBootError() {
  elements.bootPanel.classList.add("is-error");
  elements.bootBar.hidden = true;
  elements.bootDetail.textContent = "";
  elements.bootMessage.textContent =
    "Something went wrong loading the analyzer. Try refreshing the page.";
}

export function setAnalysisStatus(text, isError = false) {
  elements.status.textContent = text;
  elements.status.classList.toggle("is-error", Boolean(isError));
}

export function setProgress(label, done, total) {
  if (!total) {
    return;
  }
  setAnalysisStatus(label + " " + done + " of " + total + " sections…");
}

export function setBusy(busy) {
  elements.button.disabled = busy;
  elements.fileInput.disabled = busy;
  for (const radio of elements.domainRadios) {
    radio.disabled = busy;
  }
}

// ---------------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------------

export function resetResults() {
  elements.results.hidden = true;
  elements.spectrogramScroll.scrollLeft = 0;
  elements.graphScroll.scrollLeft = 0;
  currentResults = null;
  elements.graph.replaceChildren();
  elements.tbody.replaceChildren();
  elements.summary.textContent = "";
  elements.verdict.replaceChildren();
  elements.verdict.className = "verdict";
}

export function beginResults({ duration, domain, filename }) {
  elements.results.hidden = false;
  elements.fileHeader.replaceChildren(
    headerItem(filename),
    headerItem(formatClock(duration) + " long"),
    headerItem(domain === "YouTube" ? "From YouTube" : "Recorded by you")
  );
  elements.verdict.className = "verdict is-pending";
  elements.verdict.replaceChildren(verdictHeadline("Running the diagnosis…"));
  elements.summary.textContent = "";
}

function headerItem(text) {
  const span = document.createElement("span");
  span.className = "header-item";
  span.textContent = text;
  return span;
}

export function setSpectrogram(pngBytes) {
  if (spectrogramUrl) {
    URL.revokeObjectURL(spectrogramUrl);
  }
  spectrogramUrl = URL.createObjectURL(new Blob([pngBytes], { type: "image/png" }));
  elements.spectrogramImage.src = spectrogramUrl;
}

export function renderResults(results) {
  currentResults = results;
  relayout();
  renderVerdict(results);
  renderTable(results);
}

// ---------------------------------------------------------------------------
// Layout
// ---------------------------------------------------------------------------

function relayout() {
  const viewportWidth = elements.spectrogramScroll.clientWidth;
  if (viewportWidth === 0) {
    return;
  }

  const plotWidth = Math.max(
    elements.spectrogramImage.naturalWidth || 0,
    viewportWidth - PLOT_MARGIN_LEFT - PLOT_MARGIN_RIGHT
  );
  currentContentWidth = PLOT_MARGIN_LEFT + plotWidth + PLOT_MARGIN_RIGHT;

  // box-sizing: border-box is global, so this width INCLUDES the padding: the
  // image ends up exactly plotWidth wide, in the band the SVG plots into.
  elements.spectrogramInner.style.width = currentContentWidth + "px";
  elements.spectrogramInner.style.paddingLeft = PLOT_MARGIN_LEFT + "px";
  elements.spectrogramInner.style.paddingRight = PLOT_MARGIN_RIGHT + "px";

  elements.graph.style.width = currentContentWidth + "px";
  elements.graph.setAttribute("width", String(currentContentWidth));
  elements.graph.setAttribute("height", String(GRAPH_HEIGHT));
  elements.graph.setAttribute("viewBox", "0 0 " + currentContentWidth + " " + GRAPH_HEIGHT);

  if (currentResults) {
    renderGraph(currentResults, plotWidth);
  }
}

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
  const clamped = Math.max(0, Math.min(scrollLeft, currentContentWidth - elements.spectrogramScroll.clientWidth));
  _scrollSyncing = false;
  clearTimeout(scrollSyncTimeoutId);
  elements.spectrogramScroll.scrollLeft = clamped;
  elements.graphScroll.scrollLeft = clamped;
}

// ---------------------------------------------------------------------------
// Score graph (hand-rolled SVG - a few hundred points at most)
// ---------------------------------------------------------------------------

function renderGraph(results, plotWidth) {
  const plotTop = PLOT_MARGIN_TOP;
  const plotBottom = GRAPH_HEIGHT - PLOT_MARGIN_BOTTOM;
  const plotHeight = plotBottom - plotTop;
  const duration = results.duration_seconds || 1;
  const segments = results.segments;

  // The threshold line is part of the chart, so it has to fit in the y-range
  // even when every score sits far below it.
  const maxScore = segments.reduce((max, segment) => Math.max(max, segment.anomaly_score), 0);
  const yMax = Math.max(maxScore, results.threshold) * HEADROOM || 1;

  const timeToX = (time) => PLOT_MARGIN_LEFT + (time / duration) * plotWidth;
  const scoreToY = (score) => plotBottom - (Math.min(score, yMax) / yMax) * plotHeight;

  const children = [];

  for (const tick of niceTicks(yMax, Y_TICK_TARGET)) {
    const y = scoreToY(tick);
    children.push(
      line(PLOT_MARGIN_LEFT, y, PLOT_MARGIN_LEFT + plotWidth, y, "gridline"),
      svgText(PLOT_MARGIN_LEFT - 8, y + 4, formatTick(tick), "axis-label axis-label-y")
    );
  }

  const points = segments.map((segment) => ({
    x: timeToX(segment.start_time),
    y: scoreToY(segment.anomaly_score),
    isAnomalous: segment.is_anomalous,
  }));

  for (const run of splitIntoVerdictRuns(points)) {
    children.push(
      polygon(
        [
          ...run.points.map((point) => round(point.x) + "," + round(point.y)),
          round(run.endX) + "," + round(plotBottom),
          round(run.startX) + "," + round(plotBottom),
        ],
        run.isAnomalous ? "area area-anomaly" : "area area-healthy"
      )
    );
  }

  if (points.length > 0) {
    const polyline = document.createElementNS(SVG_NS, "polyline");
    polyline.setAttribute("class", "score-line");
    polyline.setAttribute("points", points.map((point) => round(point.x) + "," + round(point.y)).join(" "));
    children.push(polyline);
  }

  const thresholdY = scoreToY(results.threshold);
  children.push(
    line(PLOT_MARGIN_LEFT, thresholdY, PLOT_MARGIN_LEFT + plotWidth, thresholdY, "threshold-line"),
    svgText(PLOT_MARGIN_LEFT + plotWidth + 6, thresholdY + 4, "unusual", "axis-label threshold-label")
  );

  children.push(line(PLOT_MARGIN_LEFT, plotBottom, PLOT_MARGIN_LEFT + plotWidth, plotBottom, "axis-line"));
  for (const tick of xTicks(duration, plotWidth)) {
    const x = timeToX(tick);
    children.push(
      line(x, plotBottom, x, plotBottom + 4, "axis-line"),
      svgText(x, plotBottom + 16, formatTick(tick) + "s", "axis-label axis-label-x")
    );
  }

  points.forEach((point, index) => {
    const segment = segments[index];
    const left = index === 0 ? point.x : (points[index - 1].x + point.x) / 2;
    const right = index === points.length - 1 ? point.x : (point.x + points[index + 1].x) / 2;

    const dot = document.createElementNS(SVG_NS, "circle");
    dot.setAttribute("class", "dot" + (segment.is_anomalous ? " is-anomaly" : ""));
    dot.setAttribute("cx", round(point.x));
    dot.setAttribute("cy", round(point.y));
    dot.setAttribute("r", "2");
    children.push(dot);

    const band = document.createElementNS(SVG_NS, "rect");
    band.setAttribute("class", "hover-band");
    band.setAttribute("x", round(left));
    band.setAttribute("y", String(plotTop));
    band.setAttribute("width", round(Math.max(right - left, 1)));
    band.setAttribute("height", String(plotHeight));
    const title = document.createElementNS(SVG_NS, "title");
    title.textContent =
      formatClock(segment.start_time) + " - " +
      (segment.is_anomalous ? "sounded unusual" : "sounded normal");
    band.appendChild(title);
    children.push(band);
  });

  elements.graph.replaceChildren(...children);
}

// Consecutive points sharing a verdict, each run extended to the midpoint it
// shares with its neighbours so adjacent fills meet exactly.
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

// ~targetCount ticks at a "nice" round step, choosing the candidate closest to
// targetCount rather than always rounding up.
function niceTicks(yMax, targetCount) {
  const magnitude = 10 ** Math.floor(Math.log10(yMax / targetCount));
  const niceStep = [1, 2, 2.5, 5, 10]
    .map((multiple) => multiple * magnitude)
    .reduce((best, step) => (Math.abs(yMax / step - targetCount) < Math.abs(yMax / best - targetCount) ? step : best));

  const ticks = [];
  for (let index = 0; index * niceStep <= yMax + niceStep * 1e-6; index += 1) {
    ticks.push(index * niceStep);
  }
  return ticks;
}

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
// Summary + table
// ---------------------------------------------------------------------------

function formatPercent(value) {
  return Math.round(value * 100) + "%";
}

const HEALTHY_EXPLANATION =
  "No unusual patterns were found in your recording. This doesn't rule out all " +
  "faults. If you still have concerns, consult a mechanic.";

const ANOMALY_EXPLANATION =
  "The analyzer found sections that sound different from healthy engine " +
  "recordings. This is not a confirmed fault, please consult a mechanic for a " +
  "proper diagnosis.";

function renderVerdict(results) {
  const anomalousCount = results.segments.filter((segment) => segment.is_anomalous).length;
  const isAnomaly = results.overall_verdict === "anomaly";

  elements.verdict.className = "verdict " + (isAnomaly ? "is-anomaly" : "is-healthy");
  elements.verdict.replaceChildren(
    verdictHeadline(isAnomaly ? "⚠ Unusual sounds detected" : "✓ Your engine sounds healthy"),
    verdictExplanation(isAnomaly ? ANOMALY_EXPLANATION : HEALTHY_EXPLANATION)
  );

  elements.summary.textContent =
    (anomalousCount === 0
      ? "None of the " + results.segments.length + " sections sounded unusual"
      : anomalousCount + " of " + results.segments.length + " sections sounded unusual") +
    " · Certainty: " + formatPercent(results.overall_confidence);
}

function verdictHeadline(text) {
  const heading = document.createElement("p");
  heading.className = "verdict-headline";
  heading.textContent = text;
  return heading;
}

function verdictExplanation(text) {
  const paragraph = document.createElement("p");
  paragraph.className = "verdict-explanation";
  paragraph.textContent = text;
  return paragraph;
}

function renderTable(results) {
  elements.tbody.replaceChildren(...results.segments.map((segment) => buildSegmentRow(segment)));
}

function buildSegmentRow(segment) {
  const row = document.createElement("tr");
  if (segment.is_anomalous) {
    row.className = "is-anomaly";
  }

  row.append(
    cell(String(segment.index + 1)),
    cell(formatClock(segment.start_time)),
    cell(formatClock(segment.end_time)),
    verdictCell(segment.is_anomalous),
    cell(formatPercent(segment.confidence))
  );

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
  td.className = isAnomalous ? "verdict-anomaly" : "verdict-healthy";
  td.textContent = isAnomalous ? "⚠ Unusual" : "✓ Normal";
  return td;
}

// m:ss - a rider reads a position in a recording, not a float.
function formatClock(seconds) {
  const whole = Math.round(seconds);
  return Math.floor(whole / 60) + ":" + String(whole % 60).padStart(2, "0");
}

function scrollToTime(time) {
  if (!currentResults) {
    return;
  }
  const plotWidth = currentContentWidth - PLOT_MARGIN_LEFT - PLOT_MARGIN_RIGHT;
  const x = PLOT_MARGIN_LEFT + (time / (currentResults.duration_seconds || 1)) * plotWidth;
  scrollBothTo(x - elements.spectrogramScroll.clientWidth / 2);
}
