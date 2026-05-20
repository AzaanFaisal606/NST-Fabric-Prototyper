export default function ProgressBar({ status }) {
  if (!status) return null;
  const pct = status.progress ?? 0;

  // label shows current pipeline stage; iteration counter only during NST
  let label = status.status;
  if (status.status === "processing" && status.stage) {
    label = status.stage === "nst"
      ? `NST — ${status.current_iter}/${status.total_iter}`
      : status.stage;
  }

  return (
    <div className="space-y-2">
      <div className="flex justify-between text-sm">
        <span>{label}</span>
        <span className="text-neutral-400">{pct.toFixed(1)}%</span>
      </div>
      <div className="w-full h-2 bg-neutral-800 rounded overflow-hidden">
        <div className="h-full bg-emerald-500 transition-[width]" style={{ width: `${pct}%` }} />
      </div>
      {status.status === "error" && status.error_message && (
        <div className="text-sm text-rose-400">{status.error_message}</div>
      )}
    </div>
  );
}
