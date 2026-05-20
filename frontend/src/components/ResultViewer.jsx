import { resultUrl } from "../api";

export default function ResultViewer({ jobId, manifest }) {
  if (!manifest) return null;
  const url = resultUrl(manifest.url);
  return (
    <div className="bg-neutral-800/50 border border-neutral-700 rounded p-4 space-y-3">
      <h3 className="text-sm font-semibold">Result</h3>
      <img src={url} className="max-w-full rounded" />
      <a
        href={url}
        download
        className="inline-block px-3 py-1 rounded bg-emerald-600 hover:bg-emerald-500 text-sm font-medium"
      >
        Download PNG
      </a>
    </div>
  );
}
