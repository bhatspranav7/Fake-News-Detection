import { Globe } from "lucide-react";
import type { AnalysisResult } from "../types";
import { pct } from "../utils";
import { SectionCard } from "./SectionCard";

interface Props {
  source: AnalysisResult["source"];
}

const CRED_COLOR: Record<string, string> = {
  high: "var(--success)",
  medium: "var(--warning)",
  low: "var(--danger)",
  unknown: "var(--text-muted)",
};

export function SourceCard({ source }: Props) {
  if (!source.domain && source.credibility === "unknown" && source.notes.length === 0) return null;
  const color = CRED_COLOR[source.credibility] ?? CRED_COLOR.unknown;

  return (
    <SectionCard title="Source credibility" subtitle={source.domain ?? "No domain — text was pasted directly"} icon={<Globe size={18} />}>
      <div className="source">
        <div className="source__cred" style={{ ["--c" as string]: color }}>
          <span className="source__cred-label">{source.credibility}</span>
          <span className="source__cred-score">{source.score !== null ? pct(source.score, 0) : "n/a"}</span>
        </div>
        {source.score !== null && (
          <div className="meter meter--lg">
            <span style={{ width: `${Math.round(source.score * 100)}%`, background: color }} />
          </div>
        )}
        {source.notes.length > 0 && (
          <ul className="notes">
            {source.notes.map((n, i) => (
              <li key={i}>{n}</li>
            ))}
          </ul>
        )}
      </div>
    </SectionCard>
  );
}
