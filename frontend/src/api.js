const BASE = "http://localhost:8000";

export async function getHealth() {
  const r = await fetch(`${BASE}/health`);
  if (!r.ok) throw new Error("health fetch failed");
  return await r.json();
}

export async function postSegment(imageFile, points) {
  const fd = new FormData();
  fd.append("image", imageFile);
  fd.append("points", JSON.stringify(points));
  const r = await fetch(`${BASE}/segment`, { method: "POST", body: fd });
  if (!r.ok) throw new Error(`segment failed: ${r.status}`);
  return await r.blob();
}

export async function postStylize({ targetImage, sourceImage, targetMask, sourceMask, ratio, iterations, suppressTargetPattern, coarseFraction, colorStrength }) {
  const fd = new FormData();
  fd.append("target_image", targetImage);
  fd.append("source_image", sourceImage);
  fd.append("target_mask", targetMask, "target_mask.png");
  fd.append("source_mask", sourceMask, "source_mask.png");
  fd.append("alpha_beta_ratio", String(ratio));
  fd.append("iterations", String(iterations));
  fd.append("suppress_target_pattern", suppressTargetPattern ? "true" : "false");
  fd.append("coarse_fraction", String(coarseFraction));
  fd.append("color_strength", String(colorStrength));
  const r = await fetch(`${BASE}/stylize`, { method: "POST", body: fd });
  if (!r.ok) throw new Error(`stylize failed: ${r.status}`);
  return (await r.json()).job_id;
}

export async function getStatus(jobId) {
  const r = await fetch(`${BASE}/status/${jobId}`);
  if (!r.ok) throw new Error("status failed");
  return await r.json();
}

export async function getManifest(jobId) {
  const r = await fetch(`${BASE}/result/${jobId}`);
  if (!r.ok) throw new Error("manifest failed");
  return await r.json();
}

export function resultUrl(path) {
  return `${BASE}${path}`;
}
