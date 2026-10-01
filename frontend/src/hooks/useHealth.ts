import { useEffect, useState } from "react";
import { api } from "../api";
import type { Health } from "../types";

export interface HealthState {
  health: Health | null;
  online: boolean;
  checking: boolean;
  lastChecked: number | null;
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
        const h = await api.health(controller.signal);
        if (!cancelled) {
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
