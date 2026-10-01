import { useState } from "react";
import { ChevronDown, ExternalLink, ListChecks } from "lucide-react";
import type { Claim } from "../types";
import { pct } from "../utils";
import { SectionCard } from "./SectionCard";

interface Props {
  claims: Claim[];
}

function ClaimRow({ claim, index }: { claim: Claim; index: number }) {
  const [open, setOpen] = useState(index === 0);
  const hasEvidence = claim.evidence.length > 0;
  return (
    <li className={`claim claim--${claim.verdict}`}>
      <button className="claim__head" onClick={() => setOpen((v) => !v)} aria-expanded={open} disabled={!hasEvidence}>
        <span className="claim__index">{index + 1}</span>
        <span className="claim__text">{claim.claim}</span>
        <span className={`badge badge--${claim.verdict}`}>{claim.verdict}</span>
        <span className="claim__conf">{pct(claim.confidence, 0)}</span>
        {hasEvidence && <ChevronDown size={16} className={`claim__chev ${open ? "claim__chev--open" : ""}`} />}
      </button>
      {hasEvidence && open && (
        <ul className="evidence">
          {claim.evidence.map((e, i) => (
            <li key={`${e.url}-${i}`} className="evidence__item">
              <div className="evidence__head">
                <a href={e.url} target="_blank" rel="noreferrer noopener" className="evidence__title">
                  {e.title || e.url}
                  <ExternalLink size={12} />
                </a>
                <span className={`badge badge--${e.stance}`}>{e.stance}</span>
              </div>
              {e.snippet && <p className="evidence__snippet">{e.snippet}</p>}
              <span className="evidence__host">{safeHost(e.url)}</span>
            </li>
          ))}
        </ul>
      )}
      {!hasEvidence && <p className="claim__none">No evidence was retrieved for this claim.</p>}
    </li>
  );
}

function safeHost(url: string): string {
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return "";
  }
}

export function ClaimsList({ claims }: Props) {
  if (!claims || claims.length === 0) return null;
  const counts = claims.reduce(
    (acc, c) => {
      acc[c.verdict] += 1;
      return acc;
    },
    { supported: 0, refuted: 0, unverified: 0 },
  );
  return (
    <SectionCard
      title="Claims & evidence"
      subtitle={`${claims.length} claim${claims.length === 1 ? "" : "s"} extracted · ${counts.supported} supported · ${counts.refuted} refuted · ${counts.unverified} unverified`}
      icon={<ListChecks size={18} />}
    >
      <ul className="claims">
        {claims.map((c, i) => (
          <ClaimRow key={`${c.claim}-${i}`} claim={c} index={i} />
        ))}
      </ul>
    </SectionCard>
  );
}
