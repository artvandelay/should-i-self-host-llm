import { useCallback, useEffect, useRef, useState } from "react";

type Primitive = string | number | boolean;

function encode(v: Primitive): string {
  return String(v);
}

function decode<T extends Primitive>(raw: string, fallback: T): T {
  if (typeof fallback === "number") {
    const n = Number(raw);
    return (Number.isFinite(n) ? n : fallback) as T;
  }
  if (typeof fallback === "boolean") return (raw === "1" || raw === "true") as T;
  return raw as T;
}

/**
 * Tiny URL-querystring–backed state. All keys live on the same `?` query string;
 * updates are debounced into a single `replaceState` per tick.
 */
export function useUrlState<T extends Primitive>(
  key: string,
  fallback: T
): [T, (v: T) => void] {
  const initial = (() => {
    if (typeof window === "undefined") return fallback;
    const params = new URLSearchParams(window.location.search);
    const raw = params.get(key);
    return raw == null ? fallback : decode(raw, fallback);
  })();

  const [value, setValue] = useState<T>(initial);
  const fallbackRef = useRef(fallback);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const params = new URLSearchParams(window.location.search);
    if (value === fallbackRef.current) {
      params.delete(key);
    } else {
      params.set(key, encode(value));
    }
    const qs = params.toString();
    const next = `${window.location.pathname}${qs ? `?${qs}` : ""}${window.location.hash}`;
    window.history.replaceState(null, "", next);
  }, [key, value]);

  const set = useCallback((v: T) => setValue(v), []);
  return [value, set];
}
