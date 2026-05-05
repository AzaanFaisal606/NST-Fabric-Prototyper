import { useEffect, useState } from "react";
import { getFonts } from "../api";

export default function FontSelector({ value, onChange }) {
  const [fonts, setFonts] = useState([]);

  useEffect(() => {
    getFonts().then(setFonts).catch(console.error);
  }, []);

  return (
    <div className="space-y-2">
      <label className="block text-sm font-medium">Font</label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full bg-neutral-800 border border-neutral-700 rounded px-2 py-1.5 text-sm"
      >
        <option value="">— select —</option>
        {fonts.map((f) => <option key={f} value={f}>{f}</option>)}
      </select>
    </div>
  );
}
