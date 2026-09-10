// Thin fetch wrappers for every endpoint api/labeling.py exposes.

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

export function spectrogramUrl(segmentId) {
  return `/spectrogram/${encodeURIComponent(segmentId)}`;
}

export async function confirmUpload(uploadId, labels) {
  return requestJson(`/uploads/${encodeURIComponent(uploadId)}/confirm`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ labels }),
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
