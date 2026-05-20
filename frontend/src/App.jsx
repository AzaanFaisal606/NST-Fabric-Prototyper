import { useEffect, useState } from "react";
import GarmentCanvas from "./components/GarmentCanvas";
import ParamSliders from "./components/ParamSliders";
import StylizeButton from "./components/StylizeButton";
import ProgressBar from "./components/ProgressBar";
import ResultViewer from "./components/ResultViewer";
import { getHealth, postStylize, getStatus, getManifest } from "./api";

export default function App() {
  // segmentation outputs (image File + mask Blob per side)
  const [target_image, set_target_image] = useState(null);
  const [target_mask, set_target_mask] = useState(null);
  const [source_image, set_source_image] = useState(null);
  const [source_mask, set_source_mask] = useState(null);

  // stylize params
  const [ratio, set_ratio] = useState(1e-4);
  const [iterations, set_iterations] = useState(500);
  const [suppress_target_pattern, set_suppress] = useState(false);
  const [coarse_fraction, set_coarse_fraction] = useState(0.4);  // §6.2 split: 0.4 -> 40% coarse / 60% fine
  const [color_strength, set_color_strength] = useState(0.5);    // 0..1 blend toward source LAB hist

  // job state
  const [job_id, set_job_id] = useState(null);
  const [status, set_status] = useState(null);
  const [manifest, set_manifest] = useState(null);
  const [submitting, set_submitting] = useState(false);

  // device label from /health
  const [device, set_device] = useState(null);
  useEffect(() => {
    getHealth().then((h) => set_device(h.device)).catch(() => set_device("unknown"));
  }, []);

  // poll /status while job is processing
  useEffect(() => {
    if (!job_id || (status && (status.status === "complete" || status.status === "error"))) return;
    const id = setInterval(async () => {
      try {
        const s = await getStatus(job_id);
        set_status(s);
        if (s.status === "complete") {
          const m = await getManifest(job_id);
          set_manifest(m);
          clearInterval(id);
        } else if (s.status === "error") {
          clearInterval(id);
        }
      } catch (e) {
        console.error(e);
      }
    }, 1500);
    return () => clearInterval(id);
  }, [job_id, status?.status]);

  async function handle_target_ready(mask_blob, image_file) {
    set_target_mask(mask_blob);
    set_target_image(image_file);
  }

  async function handle_source_ready(mask_blob, image_file) {
    set_source_mask(mask_blob);
    set_source_image(image_file);
  }

  async function handle_submit() {
    if (!target_image || !target_mask || !source_image || !source_mask) return;
    set_submitting(true);
    set_manifest(null);
    set_status({ status: "queued", progress: 0 });
    try {
      const id = await postStylize({
        targetImage: target_image,
        sourceImage: source_image,
        targetMask: target_mask,
        sourceMask: source_mask,
        ratio,
        iterations,
        suppressTargetPattern: suppress_target_pattern,
        coarseFraction: coarse_fraction,
        colorStrength: color_strength,
      });
      set_job_id(id);
    } catch (e) {
      set_status({ status: "error", error_message: String(e) });
    } finally {
      set_submitting(false);
    }
  }

  const can_submit =
    !!target_image && !!target_mask && !!source_image && !!source_mask &&
    !submitting && (status?.status !== "processing");

  return (
    <div className="min-h-screen p-6 max-w-6xl mx-auto space-y-6">
      <div className="flex items-baseline justify-between">
        <h1 className="text-2xl font-bold">NST Garment Stylizer</h1>
        <span className="text-xs text-neutral-400">
          running on {device ?? "..."}
        </span>
      </div>

      <div className="grid md:grid-cols-2 gap-4">
        <GarmentCanvas label="Target (garment to re-style)" onMaskReady={handle_target_ready} />
        <GarmentCanvas label="Source (pattern / fabric)"   onMaskReady={handle_source_ready} />
      </div>

      <div className="grid md:grid-cols-2 gap-6 bg-neutral-800/50 border border-neutral-700 rounded p-4">
        <ParamSliders
          ratio={ratio}
          iterations={iterations}
          suppressTargetPattern={suppress_target_pattern}
          coarseFraction={coarse_fraction}
          colorStrength={color_strength}
          onRatioChange={set_ratio}
          onIterChange={set_iterations}
          onSuppressChange={set_suppress}
          onCoarseFractionChange={set_coarse_fraction}
          onColorStrengthChange={set_color_strength}
        />
        <div className="space-y-4">
          <StylizeButton disabled={!can_submit} onClick={handle_submit} />
          <ProgressBar status={status} />
        </div>
      </div>

      <ResultViewer jobId={job_id} manifest={manifest} />
    </div>
  );
}
