import { Clock, FileText, Globe, Hash, ShieldAlert, ShieldCheck, ShieldQuestion } from "lucide-react";
import type { AnalysisResult } from "../types";
import { fmtDate, fmtMs, pct, truncate, verdictColor, verdictLabel } from "../utils";
import { Gauge } from "./Gauge";

interface Props {
  result: AnalysisResult;
}

function VerdictIcon({ verdict }: { verdict: AnalysisResult["verdict"] }) {
  if (verdict === "fake") return <ShieldAlert size={28} />;
  if (verdict === "real") return <ShieldCheck size={28} />;
  return <ShieldQuestion size={28} />;
}

export function VerdictCard({ result }: Props) {
  const color = verdictColor(result.verdict);
  const snippet = result.input.title || truncate(result.input.text, 220);

  return (
    <section className={`card verdict verdict--${result.verdict}`} style={{ ["--verdict" as string]: color }}>
      <div className="verdict__main">
        <div className="verdict__label-wrap">
          <span className="verdict__icon">
            <VerdictIcon verdict={result.verdict} />
          </span>
          <div>
            <div className="verdict__eyebrow">Verdict</div>
            <div className="verdict__label">{verdictLabel(result.verdict)}</div>
          </div>
        </div>

        <div className="verdict__stats">
          <div className="stat">
            <span className="stat__label">Confidence</span>
            <span className="stat__value">{pct(result.confidence, 0)}</span>
            <div className="meter">
              <span style={{ width: `${Math.round(result.confidence * 100)}%`, background: color }} />
            </div>
          </div>
          <div className="stat">
            <span className="stat__label">
              <Clock size={12} /> Latency
            </span>
            <span className="stat__value">{fmtMs(result.latency_ms)}</span>
          </div>
          <div className="stat">
            <span className="stat__label">
              <Hash size={12} /> Mode
            </span>
            <span className="stat__value stat__value--sm">
              <span className={`chip chip--${result.mode}`}>{result.mode}</span>
            </span>
          </div>
        </div>

        <div className="verdict__input">
          <div className="verdict__input-meta">
            {result.input.domain ? (
              <span className="meta">
                <Globe size={13} /> {result.input.domain}
              </span>
            ) : (
              <span className="meta">
                <FileText size={13} /> Pasted text
              </span>
            )}
            <span className="meta muted">{result.input.word_count} words</span>
            <span className="meta muted">{fmtDate(result.created_at)}</span>
          </div>
          <p className="verdict__snippet">{snippet}</p>
          {result.input.url && (
            <a className="verdict__url" href={result.input.url} target="_blank" rel="noreferrer noopener">
              {truncate(result.input.url, 80)}
            </a>
          )}
        </div>
      </div>

      <div className="verdict__gauge">
        <Gauge value={result.fake_probability} />
        <div className="verdict__gauge-caption">
          <span className="legend-dot" style={{ background: "var(--success)" }} /> real
          <span className="legend-dot" style={{ background: "var(--warning)", marginLeft: 12 }} /> uncertain
          <span className="legend-dot" style={{ background: "var(--danger)", marginLeft: 12 }} /> fake
        </div>
      </div>
    </section>
  );
}
