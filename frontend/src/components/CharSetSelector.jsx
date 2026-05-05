const PRESETS = [
  { key: "uppercase",        label: "A-Z" },
  { key: "uppercase_digits", label: "A-Z + 0-9" },
  { key: "letters",          label: "A-Z + a-z" },
  { key: "alphanumeric",     label: "A-Z + a-z + 0-9" },
  { key: "custom",           label: "Custom" },
];

export default function CharSetSelector({ charset, custom, onCharsetChange, onCustomChange }) {
  return (
    <div className="space-y-2">
      <label className="block text-sm font-medium">Characters</label>
      <select
        value={charset}
        onChange={(e) => onCharsetChange(e.target.value)}
        className="w-full bg-neutral-800 border border-neutral-700 rounded px-2 py-1.5 text-sm"
      >
        {PRESETS.map((p) => <option key={p.key} value={p.key}>{p.label}</option>)}
      </select>
      {charset === "custom" && (
        <input
          type="text"
          value={custom}
          onChange={(e) => onCustomChange(e.target.value)}
          placeholder="ABC123"
          className="w-full bg-neutral-800 border border-neutral-700 rounded px-2 py-1.5 text-sm"
        />
      )}
    </div>
  );
}
