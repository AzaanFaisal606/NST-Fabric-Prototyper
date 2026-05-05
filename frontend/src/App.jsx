import { useEffect, useState } from "react";
import StyleUploader from "./components/StyleUploader";
import FontSelector from "./components/FontSelector";
import CharSetSelector from "./components/CharSetSelector";
import ParamSliders from "./components/ParamSliders";
import StylizeButton from "./components/StylizeButton";
import ProgressBar from "./components/ProgressBar";
import ResultGrid from "./components/ResultGrid";
import { postStylize, getStatus, getManifest } from "./api";

export default function App() {
  const [styleFile, setStyleFile] = useState(null);
  const [font, setFont] = useState("");
  const [charset, setCharset] = useState("uppercase");
  const [custom, setCustom] = useState("");
  const [ratio, setRatio] = useState(1e-4);
  const [iterations, setIterations] = useState(300);

  const [jobId, setJobId] = useState(null);
  const [status, setStatus] = useState(null);
  const [manifest, setManifest] = useState(null);
  const [submitting, setSubmitting] = useState(false);

  // poll /status while job is processing
  useEffect(() => {
    if (!jobId || (status && (status.status === "complete" || status.status === "error"))) return;
    const id = setInterval(async () => {
      try {
        const s = await getStatus(jobId);
        setStatus(s);
        if (s.status === "complete") {
          const m = await getManifest(jobId);
          setManifest(m);
          clearInterval(id);
        } else if (s.status === "error") {
          clearInterval(id);
        }
      } catch (e) {
        console.error(e);
      }
    }, 1500);
    return () => clearInterval(id);
  }, [jobId, status?.status]);

  async function handleSubmit() {
    if (!styleFile || !font) return;
    setSubmitting(true);
    setManifest(null);
    setStatus({ status: "queued", progress: 0 });
    try {
      const id = await postStylize({ styleFile, font, charset, custom, ratio, iterations });
      setJobId(id);
    } catch (e) {
      setStatus({ status: "error", error_message: String(e) });
    } finally {
      setSubmitting(false);
    }
  }

  const canSubmit = !!styleFile && !!font && !submitting && (status?.status !== "processing");

  return (
    <div className="min-h-screen p-6 max-w-5xl mx-auto space-y-6">
      <h1 className="text-2xl font-bold">NST Font Stylizer</h1>

      <div className="grid md:grid-cols-2 gap-6 bg-neutral-800/50 border border-neutral-700 rounded p-4">
        <div className="space-y-4">
          <StyleUploader onChange={setStyleFile} />
          <FontSelector value={font} onChange={setFont} />
          <CharSetSelector
            charset={charset} custom={custom}
            onCharsetChange={setCharset} onCustomChange={setCustom}
          />
        </div>
        <div className="space-y-4">
          <ParamSliders
            ratio={ratio} iterations={iterations}
            onRatioChange={setRatio} onIterChange={setIterations}
          />
          <StylizeButton disabled={!canSubmit} onClick={handleSubmit} />
          <ProgressBar status={status} />
        </div>
      </div>

      <ResultGrid jobId={jobId} manifest={manifest} />
    </div>
  );
}
