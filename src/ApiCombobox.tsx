import { useEffect, useMemo, useRef, useState } from "react";
import { Command } from "cmdk";
import { ChevronDown, Check, Search } from "lucide-react";

export interface ApiOption {
  value: string;
  label: string;
  /** Optional grouping bucket (e.g. "OpenAI", "Anthropic"). */
  group?: string;
}

interface Props {
  label: string;
  value: string;
  onChange: (v: string) => void;
  options: ApiOption[];
  hint?: string;
}

/** Searchable combobox for the (potentially huge) API model picker. */
export function ApiCombobox({ label, value, onChange, options, hint }: Props) {
  const [open, setOpen] = useState(false);
  const [search, setSearch] = useState("");
  const wrapRef = useRef<HTMLDivElement>(null);

  const selected = options.find((o) => o.value === value);

  // Group options by `group` for visual sections
  const grouped = useMemo(() => {
    const groups = new Map<string, ApiOption[]>();
    for (const o of options) {
      const g = o.group ?? "Other";
      if (!groups.has(g)) groups.set(g, []);
      groups.get(g)!.push(o);
    }
    // Sort groups: Custom always last
    return [...groups.entries()].sort(([a], [b]) => {
      if (a === "Custom") return 1;
      if (b === "Custom") return -1;
      return a.localeCompare(b);
    });
  }, [options]);

  // Close on outside click
  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (wrapRef.current && !wrapRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open]);

  return (
    <div className="flex flex-col" ref={wrapRef}>
      <label className="text-sm font-medium text-slate-700 mb-1">{label}</label>
      <div className="relative">
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          className="w-full px-3 py-2 border border-slate-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-400 focus:border-indigo-400 bg-white text-left flex items-center justify-between gap-2"
        >
          <span className="truncate text-sm">
            {selected?.label ?? "Select an API…"}
          </span>
          <ChevronDown className="w-4 h-4 text-slate-400 flex-shrink-0" />
        </button>

        {open && (
          <div className="absolute z-30 mt-1 w-full bg-white border border-slate-200 rounded-md shadow-lg overflow-hidden">
            <Command shouldFilter={true} className="flex flex-col">
              <div className="flex items-center gap-2 px-3 py-2 border-b border-slate-200">
                <Search className="w-4 h-4 text-slate-400 flex-shrink-0" />
                <Command.Input
                  autoFocus
                  value={search}
                  onValueChange={setSearch}
                  placeholder="Search models, providers, prices…"
                  className="flex-1 outline-none text-sm bg-transparent"
                />
              </div>
              <Command.List className="max-h-72 overflow-y-auto p-1">
                <Command.Empty className="px-3 py-4 text-sm text-slate-500 text-center">
                  No models match "{search}".
                </Command.Empty>
                {grouped.map(([group, items]) => (
                  <Command.Group
                    key={group}
                    heading={group}
                    className="[&_[cmdk-group-heading]]:px-2 [&_[cmdk-group-heading]]:pt-2 [&_[cmdk-group-heading]]:pb-1 [&_[cmdk-group-heading]]:text-[10px] [&_[cmdk-group-heading]]:font-semibold [&_[cmdk-group-heading]]:uppercase [&_[cmdk-group-heading]]:tracking-wider [&_[cmdk-group-heading]]:text-slate-400"
                  >
                    {items.map((o) => (
                      <Command.Item
                        key={o.value}
                        value={`${group} ${o.label}`}
                        onSelect={() => {
                          onChange(o.value);
                          setOpen(false);
                          setSearch("");
                        }}
                        className="flex items-center justify-between gap-2 px-2 py-1.5 rounded text-sm cursor-pointer aria-selected:bg-indigo-50 aria-selected:text-indigo-900 data-[selected=true]:bg-indigo-50"
                      >
                        <span className="truncate">{o.label}</span>
                        {o.value === value && (
                          <Check className="w-4 h-4 text-indigo-600 flex-shrink-0" />
                        )}
                      </Command.Item>
                    ))}
                  </Command.Group>
                ))}
              </Command.List>
            </Command>
          </div>
        )}
      </div>
      {hint && <span className="text-xs text-slate-500 mt-1">{hint}</span>}
    </div>
  );
}