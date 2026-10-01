import { useState } from "react";
import { AlertTriangle, BrainCircuit, ChevronDown } from "lucide-react";
import type { AnalysisResult } from "../types";
import { SectionCard } from "./SectionCard";

interface Props {
  llm: NonNullable<AnalysisResult["llm"]>;
}

export function LlmCard({ llm }: Props) {
  const [open, setOpen] = useState(false);
  return (
    <SectionCard
      title="LLM analysis"
      subtitle={`${llm.provider} · ${llm.model}`}
      icon={<BrainCircuit size={18} />}
    >
      {llm.summary && <p className="llm__summary">{llm.summary}</p>}

      {llm.red_flags.length > 0 && (
        <div className="redflags">
          <div className="redflags__title">
            <AlertTriangle size={14} /> Red flags
          </div>
          <ul>
            {llm.red_flags.map((f, i) => (
              <li key={i}>{f}</li>
            ))}
          </ul>
        </div>
      )}

      {llm.reasoning && (
        <div className="collapsible">
          <button className="collapsible__btn" onClick={() => setOpen((v) => !v)} aria-expanded={open}>
            <ChevronDown size={16} className={`claim__chev ${open ? "claim__chev--open" : ""}`} />
            {open ? "Hide reasoning" : "Show full reasoning"}
          </button>
          {open && <pre className="llm__reasoning">{llm.reasoning}</pre>}
        </div>
      )}
    </SectionCard>
  );
}
