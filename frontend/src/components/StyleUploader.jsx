import { useState } from "react";

export default function StyleUploader({ onChange }) {
  const [preview, setPreview] = useState(null);

  function handleFile(e) {
    const f = e.target.files?.[0];
    if (!f) return;
    setPreview(URL.createObjectURL(f));
    onChange(f);
  }

  return (
    <div className="space-y-2">
      <label className="block text-sm font-medium">Style image</label>
      <input
        type="file"
        accept="image/png,image/jpeg,image/webp"
        onChange={handleFile}
        className="block w-full text-sm file:mr-3 file:px-3 file:py-1.5 file:rounded file:border-0 file:bg-neutral-700 file:text-neutral-100 hover:file:bg-neutral-600"
      />
      {preview && (
        <img src={preview} alt="style preview" className="mt-2 max-h-40 rounded border border-neutral-700" />
      )}
    </div>
  );
}
