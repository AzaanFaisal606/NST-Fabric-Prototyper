export default function ParamSliders({ ratio, iterations, onRatioChange, onIterChange }) {
  // ratio slider works in log-space: slider=0..100 -> ratio=10^(-5 + slider*4/100)
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
          type="range" min={100} max={600} step={10}
          value={iterations}
          onChange={(e) => onIterChange(Number(e.target.value))}
          className="w-full"
        />
      </div>
    </div>
  );
}
