import { useEffect, useRef, useState } from "react";
import { Check, ChevronDown, Download } from "lucide-react";
import { toCSV, toJSON, type ExportInputs } from "./exportResults";
import type { RecommendResult } from "./engine";

interface ExportMenuProps {
  inputs: ExportInputs;
  result: RecommendResult;
}

export function ExportMenu({ inputs, result }: ExportMenuProps) {
  const [open, setOpen] = useState(false);
  const [copied, setCopied] = useState<"csv" | "json" | null>(null);
  const wrapRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => {
      if (!wrapRef.current?.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, [open]);

  const copy = async (kind: "csv" | "json") => {
    const text = kind === "csv" ? toCSV(inputs, result) : toJSON(inputs, result);
    try {
      await navigator.clipboard.writeText(text);
      setCopied(kind);
      setOpen(false);
      setTimeout(() => setCopied(null), 1800);
    } catch {
      // ignore
    }
  };

  return (
    <div className="relative inline-block" ref={wrapRef}>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className={`text-sm flex items-center gap-1.5 px-3 py-2 border rounded-md transition-colors ${
          copied
            ? "border-emerald-300 bg-emerald-50 text-emerald-700"
            : "border-slate-300 hover:bg-white text-slate-700 bg-white"
        }`}
        title="Copy this comparison as CSV or JSON"
      >
        {copied ? (
          <>
            <Check className="w-4 h-4" /> Copied {copied.toUpperCase()}!
          </>
        ) : (
          <>
            <Download className="w-4 h-4" /> Export <ChevronDown className="w-3.5 h-3.5" />
          </>
        )}
      </button>
      {open && (
        <div className="absolute right-0 mt-1 w-56 bg-white border border-slate-200 rounded-md shadow-lg z-20 overflow-hidden">
          <button
            type="button"
            onClick={() => copy("csv")}
            className="w-full text-left px-3 py-2 text-sm hover:bg-indigo-50 text-slate-700"
          >
            Copy as CSV
          </button>
          <button
            type="button"
            onClick={() => copy("json")}
            className="w-full text-left px-3 py-2 text-sm hover:bg-indigo-50 text-slate-700 border-t border-slate-100"
          >
            Copy as JSON
          </button>
        </div>
      )}
    </div>
  );
}
