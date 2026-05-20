import { useEffect, useRef, useState } from "react";
import { postSegment } from "../api";

export default function GarmentCanvas({ label, onMaskReady }) {
  const [image_file, set_image_file] = useState(null);
  const [image_url, set_image_url] = useState(null);
  const [image_dims, set_image_dims] = useState(null);
  const [points, set_points] = useState([]);
  const [mask_url, set_mask_url] = useState(null);
  const [mask_blob, set_mask_blob] = useState(null);
  const [loading, set_loading] = useState(false);
  const img_ref = useRef(null);

  // revoke object URLs on unmount only (manual revocation handles in-place replacement)
  useEffect(() => {
    return () => {
      if (image_url) URL.revokeObjectURL(image_url);
      if (mask_url) URL.revokeObjectURL(mask_url);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function handle_file_change(e) {
    const f = e.target.files?.[0];
    if (!f) return;
    if (image_url) URL.revokeObjectURL(image_url);
    if (mask_url) URL.revokeObjectURL(mask_url);
    set_image_file(f);
    set_image_url(URL.createObjectURL(f));
    set_image_dims(null);
    set_points([]);
    set_mask_url(null);
    set_mask_blob(null);
  }

  function handle_load() {
    const el = img_ref.current;
    if (el) set_image_dims({ w: el.naturalWidth, h: el.naturalHeight });
  }

  async function handle_click(e) {
    if (!image_file || !image_dims) return;
    const el = img_ref.current;
    const x = Math.round((e.nativeEvent.offsetX / el.clientWidth) * image_dims.w);
    const y = Math.round((e.nativeEvent.offsetY / el.clientHeight) * image_dims.h);
    const new_pt = { x, y, label: e.shiftKey ? 0 : 1 };
    const next = [...points, new_pt];
    set_points(next);
    set_loading(true);
    try {
      const blob = await postSegment(image_file, next);
      if (mask_url) URL.revokeObjectURL(mask_url);
      set_mask_blob(blob);
      set_mask_url(URL.createObjectURL(blob));
    } catch (err) {
      console.error(err);
    } finally {
      set_loading(false);
    }
  }

  function handle_reset() {
    set_points([]);
    if (mask_url) URL.revokeObjectURL(mask_url);
    set_mask_url(null);
    set_mask_blob(null);
  }

  function handle_confirm() {
    if (mask_blob && image_file) onMaskReady(mask_blob, image_file);
  }

  const pos = points.filter((p) => p.label === 1).length;
  const neg = points.filter((p) => p.label === 0).length;

  return (
    <div className="bg-neutral-800/50 border border-neutral-700 rounded p-4 space-y-2">
      <h3 className="text-sm font-semibold">{label}</h3>
      <input
        type="file"
        accept="image/*"
        onChange={handle_file_change}
        className="text-sm text-neutral-300 file:mr-2 file:px-3 file:py-1 file:rounded file:border-0 file:bg-neutral-700 file:text-neutral-100 hover:file:bg-neutral-600"
      />
      {image_url && (
        <div className="relative inline-block max-w-full">
          <img
            ref={img_ref}
            src={image_url}
            onClick={handle_click}
            onLoad={handle_load}
            className="max-w-full max-h-[400px] cursor-crosshair select-none rounded"
            draggable={false}
          />
          {mask_url && (
            <div
              className="absolute inset-0 w-full h-full pointer-events-none rounded"
              style={{
                backgroundColor: "rgba(16, 185, 129, 0.65)",
                WebkitMaskImage: `url(${mask_url})`,
                maskImage: `url(${mask_url})`,
                WebkitMaskSize: "100% 100%",
                maskSize: "100% 100%",
                WebkitMaskRepeat: "no-repeat",
                maskRepeat: "no-repeat",
                WebkitMaskMode: "luminance",
                maskMode: "luminance",
              }}
            />
          )}
        </div>
      )}
      <div className="text-xs text-neutral-400">
        Click on garment (Shift+click = exclude). +{pos} -{neg} points.
      </div>
      <div className="flex gap-2">
        <button
          onClick={handle_reset}
          className="px-3 py-1 rounded bg-neutral-700 hover:bg-neutral-600 text-sm"
        >
          Reset
        </button>
        <button
          onClick={handle_confirm}
          disabled={!mask_blob}
          className="px-3 py-1 rounded bg-emerald-600 hover:bg-emerald-500 disabled:bg-neutral-700 disabled:cursor-not-allowed text-sm font-medium"
        >
          Use mask
        </button>
      </div>
      {loading && <div className="text-xs text-neutral-400">Segmenting...</div>}
    </div>
  );
}
