const BASE = "http://localhost:8000";

export async function getFonts() {
  const r = await fetch(`${BASE}/fonts`);
  if (!r.ok) throw new Error("fonts fetch failed");
  return (await r.json()).fonts;
}

export async function postStylize({ styleFile, font, charset, custom, ratio, iterations }) {
  const fd = new FormData();
  fd.append("style_image", styleFile);
  fd.append("font", font);
  fd.append("charset", charset);
  fd.append("custom", custom);
  fd.append("alpha_beta_ratio", String(ratio));
  fd.append("iterations", String(iterations));
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

export function zipUrl(jobId) {
  return `${BASE}/result/${jobId}/zip`;
}
