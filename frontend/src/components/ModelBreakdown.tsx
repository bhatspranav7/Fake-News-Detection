import { Layers } from "lucide-react";
import type { AnalysisResult } from "../types";
import { modelLabel, pct, probColor } from "../utils";
import { SectionCard } from "./SectionCard";

interface Props {
  ml: AnalysisResult["ml"];
}

interface Row {
  key: string;
  label: string;
  value: number;
  highlight?: boolean;
}

export function ModelBreakdown({ ml }: Props) {
  const rows: Row[] = Object.entries(ml.models)
    .sort((a, b) => b[1] - a[1])
    .map(([key, value]) => ({ key, label: modelLabel(key), value }));
  rows.push({ key: "ensemble", label: "Stacked ensemble (meta-learner)", value: ml.ensemble_prob, highlight: true });

  return (
    <SectionCard
      title="Model breakdown"
      subtitle="Fake probability from each base model and the stacked ensemble"
      icon={<Layers size={18} />}
    >
      <ul className="bars">
        {rows.map((r, i) => (
          <li key={r.key} className={`bars__row ${r.highlight ? "bars__row--highlight" : ""}`} style={{ animationDelay: `${i * 60}ms` }}>
            <div className="bars__label">
              <span>{r.label}</span>
              <span className="bars__value" style={{ color: probColor(r.value) }}>
                {pct(r.value, 1)}
              </span>
            </div>
            <div className="bars__track">
              <span className="bars__fill" style={{ width: `${Math.round(r.value * 100)}%`, background: probColor(r.value) }} />
              <span className="bars__mid" />
            </div>
          </li>
        ))}
      </ul>
      <p className="hint">Values above 50% lean fake; the midline marks the decision boundary.</p>
    </SectionCard>
  );
}
