// Trigger a retrain, poll GET /train/{job_id} until it settles, render the
// rolling log tail and (on success) the metrics panel.

import { startTraining, getTrainingJob } from "./api.js";

const epochsInput = document.getElementById("epochs-input");
const runNameInput = document.getElementById("run-name-input");
const retrainButton = document.getElementById("retrain-button");
const trainStatus = document.getElementById("train-status");
const trainLog = document.getElementById("train-log");
const trainMetricsPanel = document.getElementById("train-metrics");
const trainMetricsList = document.getElementById("train-metrics-list");

const POLL_INTERVAL_MS = 2000;
let pollTimer = null;

export function initTrainTab() {
  retrainButton.addEventListener("click", handleRetrain);
}

async function handleRetrain() {
  const epochs = Number.parseInt(epochsInput.value, 10) || 10;
  const runName = runNameInput.value.trim();

  retrainButton.disabled = true;
  trainMetricsPanel.hidden = true;
  trainMetricsList.innerHTML = "";
  trainLog.textContent = "";
  trainStatus.textContent = "Starting…";

  try {
    const job = await startTraining(epochs, runName);
    renderJob(job);
    pollJob(job.job_id);
  } catch (error) {
    trainStatus.textContent = `Could not start training: ${error.message}`;
    retrainButton.disabled = false;
  }
}

function pollJob(jobId) {
  window.clearInterval(pollTimer);
  pollTimer = window.setInterval(async () => {
    try {
      const job = await getTrainingJob(jobId);
      renderJob(job);
      if (job.status !== "running") {
        window.clearInterval(pollTimer);
        retrainButton.disabled = false;
      }
    } catch (error) {
      trainStatus.textContent = `Lost track of job: ${error.message}`;
      window.clearInterval(pollTimer);
      retrainButton.disabled = false;
    }
  }, POLL_INTERVAL_MS);
}

function renderJob(job) {
  trainLog.textContent = job.log_tail;
  trainLog.scrollTop = trainLog.scrollHeight;

  if (job.status === "running") {
    trainStatus.textContent = `Job ${job.job_id} running…`;
  } else if (job.status === "completed") {
    trainStatus.textContent = `Job ${job.job_id} completed (run ${job.run_id ?? "unknown"}).`;
    renderMetrics(job.metrics);
  } else {
    trainStatus.textContent = `Job ${job.job_id} failed: ${job.error ?? "see log"}`;
  }
}

function renderMetrics(metrics) {
  trainMetricsList.innerHTML = "";
  if (!metrics) {
    return;
  }
  for (const [key, value] of Object.entries(metrics)) {
    const dt = document.createElement("dt");
    dt.textContent = key;
    const dd = document.createElement("dd");
    dd.textContent = typeof value === "number" ? value.toFixed(6) : String(value);
    trainMetricsList.appendChild(dt);
    trainMetricsList.appendChild(dd);
  }
  trainMetricsPanel.hidden = false;
}
