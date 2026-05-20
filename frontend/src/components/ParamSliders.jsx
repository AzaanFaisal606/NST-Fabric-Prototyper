export default function ParamSliders({
  ratio, iterations, suppressTargetPattern, coarseFraction, colorStrength,
  onRatioChange, onIterChange, onSuppressChange, onCoarseFractionChange, onColorStrengthChange,
}) {
  // log-scale slider: 0..100 -> ratio 1e-5..1e-1
  const sliderVal = ((Math.log10(ratio) + 5) / 4) * 100;

  function setFromSlider(v) {
    const r = Math.pow(10, -5 + (v / 100) * 4);
    onRatioChange(r);
  }

  // split readout: e.g. iterations=500, coarseFraction=0.4 -> "200 / 300"
  const coarseIters = Math.max(1, Math.round(iterations * coarseFraction));
  const fineIters = Math.max(1, iterations - coarseIters);

  return (
    <div className="space-y-4">
      {/* ==== Slider: Style ↔ Content (α/β ratio)  (README same heading) ==== */}
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
      {/* ==== Slider: Iterations  (README same heading) ==== */}
      <div>
        <div className="flex justify-between text-sm">
          <span>Iterations (total)</span>
          <span className="text-neutral-400">{iterations}</span>
        </div>
        <input
          type="range" min={100} max={1000} step={50}
          value={iterations}
          onChange={(e) => onIterChange(Number(e.target.value))}
          className="w-full"
        />
      </div>
      {/* ==== Slider: Coarse / fine split  (README: Refinements → Coarse → fine split) ==== */}
      <div>
        <div className="flex justify-between text-sm">
          <span>Coarse / fine split</span>
          <span className="text-neutral-400">{coarseIters} @ 384 / {fineIters} @ 768</span>
        </div>
        <input
          type="range" min={10} max={90} step={5}
          value={Math.round(coarseFraction * 100)}
          onChange={(e) => onCoarseFractionChange(Number(e.target.value) / 100)}
          className="w-full"
        />
        <div className="flex justify-between text-xs text-neutral-500">
          <span>more fine (detail / colour)</span><span>more coarse (macro patterns)</span>
        </div>
      </div>
      {/* ==== Slider: Colour strength  (README: Refinements → Colour strength tuning) ==== */}
      <div>
        <div className="flex justify-between text-sm">
          <span>Color strength</span>
          <span className="text-neutral-400">{Math.round(colorStrength * 100)}%</span>
        </div>
        <input
          type="range" min={0} max={100} step={5}
          value={Math.round(colorStrength * 100)}
          onChange={(e) => onColorStrengthChange(Number(e.target.value) / 100)}
          className="w-full"
        />
        <div className="flex justify-between text-xs text-neutral-500">
          <span>muted (NST natural)</span><span>vivid (force-match source)</span>
        </div>
      </div>
      {/* ==== Toggle: Suppress target pattern  (README same heading) ==== */}
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
