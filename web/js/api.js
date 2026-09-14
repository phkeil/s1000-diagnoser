// Thin fetch wrappers for every endpoint the tool app exposes - api/labeling.py's
// (upload/spectrogram/confirm/train) and api/inference_tab.py's read-only
// /diagnose/* trio. Those two routers stay fully decoupled server-side; sharing
// one HTTP helper here is purely so error handling is written once.

async function readErrorDetail(response) {
  try {
    const body = await response.json();
    return body.detail || response.statusText;
  } catch {
    return response.statusText;
  }
}

async function requestJson(input, init) {
  const response = await fetch(input, init);
  if (!response.ok) {
    throw new Error(`${response.status}: ${await readErrorDetail(response)}`);
  }
  return response.json();
}

// metadata: plain object of optional recording-details form fields (contributor,
// recording_device, exhaust_system, model_year, kilometers_on_bike, oil_type,
// kilometers_since_last_oilchange, known_issues, notes). Only non-empty values
// are sent, so an omitted field reaches the server as its true default (None)
// rather than an empty string. original_codec is never sent - the server
// derives it from the upload's own file extension.
export async function uploadFile(file, domain, metadata = {}) {
  const form = new FormData();
  form.append("file", file);
  if (domain) {
    form.append("domain", domain);
  }
  for (const [key, value] of Object.entries(metadata)) {
    if (value !== null && value !== undefined && value !== "") {
      form.append(key, value);
    }
  }
  return requestJson("/uploads", { method: "POST", body: form });
}

export async function getUpload(uploadId) {
  return requestJson(`/uploads/${encodeURIComponent(uploadId)}`);
}

export function fullSpectrogramUrl(uploadId) {
  return `/uploads/${encodeURIComponent(uploadId)}/spectrogram`;
}

// regions: [{start_time, end_time, label: "healthy" | "defective" | "skip"}]
export async function confirmUpload(uploadId, regions) {
  return requestJson(`/uploads/${encodeURIComponent(uploadId)}/confirm`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ regions }),
  });
}

export async function startTraining(epochs, runName) {
  return requestJson("/train", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ epochs, run_name: runName }),
  });
}

export async function getTrainingJob(jobId) {
  return requestJson(`/train/${encodeURIComponent(jobId)}`);
}

// --- Diagnose tab (api/inference_tab.py) - read-only, never writes to the
// manifest. Deliberately a separate upload endpoint from uploadFile() above:
// the server keeps the two session stores apart, so an upload_id from one is
// meaningless to the other.

export async function uploadForDiagnosis(file, domain) {
  const form = new FormData();
  form.append("file", file);
  if (domain) {
    form.append("domain", domain);
  }
  return requestJson("/diagnose/uploads", { method: "POST", body: form });
}

export async function getDiagnoseResults(uploadId) {
  return requestJson(`/diagnose/${encodeURIComponent(uploadId)}/results`);
}

export function diagnoseSpectrogramUrl(uploadId) {
  return `/diagnose/${encodeURIComponent(uploadId)}/spectrogram`;
}
