import { useEffect, useState } from "react";
import { api } from "../api";
import type { Health } from "../types";

export interface HealthState {
  health: Health | null;
  online: boolean;
  checking: boolean;
  lastChecked: number | null;
}

/** Fill in any fields a partial / older backend might omit so the UI never crashes. */
function normalizeHealth(raw: Partial<Health> | null): Health {
  const r = raw ?? {};
  const llm = (r.llm ?? {}) as Partial<Health["llm"]>;
  const index = (r.index ?? {}) as Partial<Health["index"]>;
  return {
    status: typeof r.status === "string" ? r.status : "ok",
    version: typeof r.version === "string" ? r.version : "?",
    models_loaded: Array.isArray(r.models_loaded) ? r.models_loaded : [],
    llm: {
      provider: typeof llm.provider === "string" ? llm.provider : "unknown",
      model: typeof llm.model === "string" ? llm.model : "unknown",
      ok: llm.ok === true,
    },
    index: { analyses: typeof index.analyses === "number" ? index.analyses : 0 },
  };
}

/** Polls GET /health on an interval (default 30s). */
export function useHealth(intervalMs = 30_000): HealthState {
  const [state, setState] = useState<HealthState>({
    health: null,
    online: false,
    checking: true,
    lastChecked: null,
  });

  useEffect(() => {
    let cancelled = false;
    let controller: AbortController | null = null;

    const tick = async () => {
      controller?.abort();
      controller = new AbortController();
      try {
        const raw = (await api.health(controller.signal)) as Partial<Health> | null;
        if (!cancelled) {
          const h = normalizeHealth(raw);
          setState({ health: h, online: h.status === "ok", checking: false, lastChecked: Date.now() });
        }
      } catch {
        if (!cancelled) {
          setState((s) => ({ ...s, online: false, checking: false, lastChecked: Date.now() }));
        }
      }
    };

    void tick();
    const id = window.setInterval(() => void tick(), intervalMs);
    return () => {
      cancelled = true;
      controller?.abort();
      window.clearInterval(id);
    };
  }, [intervalMs]);

  return state;
}
