// Tab switching: Label | Train | Diagnose (Phase 3 adds Crawl).

import { initUploadTab } from "./upload.js";
import { initTrainTab } from "./train.js";
import { initDiagnoseTab } from "./diagnose.js";

const tabButtons = document.querySelectorAll(".tab-button");
const panels = {
  label: document.getElementById("tab-label"),
  train: document.getElementById("tab-train"),
  diagnose: document.getElementById("tab-diagnose"),
};

function activateTab(tabName) {
  for (const button of tabButtons) {
    button.classList.toggle("active", button.dataset.tab === tabName);
  }
  for (const [name, panel] of Object.entries(panels)) {
    panel.hidden = name !== tabName;
  }
  // The graph's content width is derived from the panel's clientWidth, which
  // reads 0 while an ancestor is still [hidden] - so re-measure on the way in.
  if (tabName === "diagnose") {
    notifyDiagnoseShown();
  }
}

for (const button of tabButtons) {
  button.addEventListener("click", () => activateTab(button.dataset.tab));
}

initUploadTab();
initTrainTab();
// Bound before the first activateTab() call below, which reads it.
const { onTabShown: notifyDiagnoseShown } = initDiagnoseTab();

activateTab("label");
