export default function ProgressBar({ status }) {
  if (!status) return null;
  const pct = status.progress ?? 0;
  return (
    <div className="space-y-2">
      <div className="flex justify-between text-sm">
        <span>
          {status.status === "processing" && status.current_char
            ? `Stylizing '${status.current_char}' — ${status.current_iter}/${status.total_iter}`
            : status.status}
        </span>
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
