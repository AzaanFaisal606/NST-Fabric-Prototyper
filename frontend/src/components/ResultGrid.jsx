import { resultUrl, zipUrl } from "../api";

export default function ResultGrid({ jobId, manifest }) {
  if (!manifest) return null;
  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold">Stylized glyphs ({manifest.chars.length})</h2>
        <a
          href={zipUrl(jobId)}
          className="text-sm px-3 py-1.5 rounded bg-neutral-800 hover:bg-neutral-700"
        >
          Download all (zip)
        </a>
      </div>
      <div className="grid grid-cols-4 sm:grid-cols-6 md:grid-cols-8 gap-2">
        {manifest.chars.map((c, i) => (
          <div key={i} className="bg-neutral-800 rounded p-1">
            <img src={resultUrl(manifest.urls[i])} alt={c} className="w-full aspect-square object-contain" />
            <div className="text-xs text-center text-neutral-400 mt-1">{c}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
