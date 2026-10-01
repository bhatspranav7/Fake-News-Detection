import { Flag, Sparkles } from "lucide-react";
import type { TopToken } from "../types";
import { SectionCard } from "./SectionCard";

interface Props {
  tokens: TopToken[];
  styleFlags: string[];
}

export function TokenChips({ tokens, styleFlags }: Props) {
  if (tokens.length === 0 && styleFlags.length === 0) return null;
  const maxW = Math.max(...tokens.map((t) => Math.abs(t.weight)), 1e-6);
  const sorted = [...tokens].sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight));

  return (
    <SectionCard
      title="Why the model thinks so"
      subtitle="Most influential tokens — red pushes toward fake, green toward real"
      icon={<Sparkles size={18} />}
    >
      {sorted.length > 0 && (
        <div className="tokens">
          {sorted.map((t, i) => {
            const intensity = Math.abs(t.weight) / maxW;
            const toward = t.weight >= 0 ? "fake" : "real";
            return (
              <span
                key={`${t.token}-${i}`}
                className={`token token--${toward}`}
                style={{ ["--i" as string]: (0.25 + intensity * 0.75).toFixed(2), animationDelay: `${i * 25}ms` }}
                title={`${t.weight >= 0 ? "+" : ""}${t.weight.toFixed(3)} toward ${toward}`}
              >
                {t.token}
                <small>{t.weight >= 0 ? "+" : "−"}{Math.abs(t.weight).toFixed(2)}</small>
              </span>
            );
          })}
        </div>
      )}
      {styleFlags.length > 0 && (
        <div className="flags">
          <div className="flags__title">
            <Flag size={14} /> Style signals
          </div>
          <ul className="flags__list">
            {styleFlags.map((f) => (
              <li key={f}>{f}</li>
            ))}
          </ul>
        </div>
      )}
    </SectionCard>
  );
}
