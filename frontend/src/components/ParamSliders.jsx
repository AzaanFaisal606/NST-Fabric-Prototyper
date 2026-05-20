export default function ParamSliders({
  ratio, iterations, suppressTargetPattern,
  onRatioChange, onIterChange, onSuppressChange,
}) {
  // log-scale slider: 0..100 -> ratio 1e-5..1e-1
  const sliderVal = ((Math.log10(ratio) + 5) / 4) * 100;

  function setFromSlider(v) {
    const r = Math.pow(10, -5 + (v / 100) * 4);
    onRatioChange(r);
  }

  return (
    <div className="space-y-4">
      <div>
        <div className="flex justify-between text-sm">
          <span>Style ↔ Content (α/β ratio)</span>
          <span className="text-neutral-400">{ratio.toExponential(1)}</span>
        </div>
        <input
          type="range" min={0} max={100} step={1}
          value={sliderVal}
          onChange={(e) => setFromSlider(Number(e.target.value))}
          className="w-full"
        />
        <div className="flex justify-between text-xs text-neutral-500">
          <span>more style</span><span>more content</span>
        </div>
      </div>
      <div>
        <div className="flex justify-between text-sm">
          <span>Iterations</span>
          <span className="text-neutral-400">{iterations}</span>
        </div>
        <input
          type="range" min={100} max={1000} step={50}
          value={iterations}
          onChange={(e) => onIterChange(Number(e.target.value))}
          className="w-full"
        />
      </div>
      <div className="flex items-center gap-2 text-sm">
        <input
          id="suppress"
          type="checkbox"
          checked={suppressTargetPattern}
          onChange={(e) => onSuppressChange(e.target.checked)}
          className="accent-emerald-500"
        />
        <label htmlFor="suppress" className="cursor-pointer">
          Target garment has a pattern (suppress before stylization)
        </label>
      </div>
    </div>
  );
}
