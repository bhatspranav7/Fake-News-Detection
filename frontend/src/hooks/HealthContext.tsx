import { createContext, useContext, type ReactNode } from "react";
import { useHealth, type HealthState } from "./useHealth";

const HealthContext = createContext<HealthState>({
  health: null,
  online: false,
  checking: true,
  lastChecked: null,
});

export function HealthProvider({ children }: { children: ReactNode }) {
  const state = useHealth(30_000);
  return <HealthContext.Provider value={state}>{children}</HealthContext.Provider>;
}

export function useHealthContext(): HealthState {
  return useContext(HealthContext);
}
