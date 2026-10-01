import { AlertCircle, Check, Circle, Loader2, MinusCircle } from "lucide-react";
import type { AgentStatus } from "../types";
import { agentLabel, fmtMs } from "../utils";

export interface TimelineStep {
  agent: string;
  status: AgentStatus | "pending";
  summary: string;
  elapsed_ms: number | null;
}

interface Props {
  steps: TimelineStep[];
  compact?: boolean;
}

function StatusIcon({ status }: { status: TimelineStep["status"] }) {
  switch (status) {
    case "running":
      return <Loader2 size={16} className="spin" />;
    case "done":
      return <Check size={16} strokeWidth={3} />;
    case "error":
      return <AlertCircle size={16} />;
    case "skipped":
      return <MinusCircle size={16} />;
    default:
      return <Circle size={10} />;
  }
}

export function AgentTimeline({ steps, compact = false }: Props) {
  return (
    <ol className={`timeline ${compact ? "timeline--compact" : ""}`}>
      {steps.map((s, i) => (
        <li key={s.agent} className={`timeline__item timeline__item--${s.status}`} style={{ animationDelay: `${i * 40}ms` }}>
          <div className="timeline__rail">
            <span className="timeline__dot">
              <StatusIcon status={s.status} />
            </span>
            {i < steps.length - 1 && <span className="timeline__line" />}
          </div>
          <div className="timeline__body">
            <div className="timeline__head">
              <span className="timeline__name">{agentLabel(s.agent)}</span>
              {s.elapsed_ms !== null && s.status !== "running" && (
                <span className="timeline__ms">{fmtMs(s.elapsed_ms)}</span>
              )}
              {s.status === "running" && <span className="timeline__ms muted">working…</span>}
            </div>
            {s.summary && <p className="timeline__summary">{s.summary}</p>}
          </div>
        </li>
      ))}
    </ol>
  );
}
