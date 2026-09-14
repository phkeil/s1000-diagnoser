// Tab switching: Label | Train (Phase 3 adds Crawl).

import { initUploadTab } from "./upload.js";
import { initTrainTab } from "./train.js";

const tabButtons = document.querySelectorAll(".tab-button");
const panels = {
  label: document.getElementById("tab-label"),
  train: document.getElementById("tab-train"),
};

function activateTab(tabName) {
  for (const button of tabButtons) {
    button.classList.toggle("active", button.dataset.tab === tabName);
  }
  for (const [name, panel] of Object.entries(panels)) {
    panel.hidden = name !== tabName;
  }
}

for (const button of tabButtons) {
  button.addEventListener("click", () => activateTab(button.dataset.tab));
}

activateTab("label");
initUploadTab();
initTrainTab();
