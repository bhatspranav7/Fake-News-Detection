import type { Verdict } from "./types";

export const MODEL_LABELS: Record<string, string> = {
  tfidf_lr: "TF-IDF + Logistic Regression",
  tfidf: "TF-IDF + Logistic Regression",
  embed_xgb: "MiniLM embeddings + XGBoost",
  embed: "MiniLM embeddings + XGBoost",
  distilbert: "DistilBERT (fine-tuned)",
  transformer: "DistilBERT (fine-tuned)",
  ensemble: "Stacked ensemble",
};

export function modelLabel(key: string): string {
  if (MODEL_LABELS[key]) return MODEL_LABELS[key];
  return key
    .split(/[_-]/)
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
}

export const AGENT_LABELS: Record<string, string> = {
  ingest: "Ingest",
  classifier: "Classifier",
  claim_extractor: "Claim extractor",
  evidence_retriever: "Evidence retriever",
  fact_checker: "Fact checker",
  source_credibility: "Source credibility",
  judge: "Judge",
};

export const AGENT_ORDER = [
  "ingest",
  "classifier",
  "claim_extractor",
  "evidence_retriever",
  "fact_checker",
  "source_credibility",
  "judge",
];

export function agentLabel(name: string): string {
  return AGENT_LABELS[name] ?? modelLabel(name);
}

export const DATASET_LABELS: Record<string, string> = {
  liar: "LIAR",
  politifact: "PolitiFact",
  gossipcop: "GossipCop",
  fakenewsnet: "FakeNewsNet",
};

export function datasetLabel(key: string): string {
  return DATASET_LABELS[key.toLowerCase()] ?? modelLabel(key);
}

export function pct(v: number | null | undefined, digits = 1): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return `${(v * 100).toFixed(digits)}%`;
}

export function fmtMs(ms: number | null | undefined): string {
  if (ms === null || ms === undefined) return "—";
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(ms < 10000 ? 2 : 1)} s`;
}

export function fmtNumber(n: number | null | undefined): string {
  if (n === null || n === undefined) return "—";
  return new Intl.NumberFormat("en-US").format(n);
}

export function fmtDate(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function truncate(s: string, n = 140): string {
  if (s.length <= n) return s;
  return `${s.slice(0, n - 1).trimEnd()}…`;
}

export function verdictColor(v: Verdict | string): string {
  switch (v) {
    case "fake":
      return "var(--danger)";
    case "real":
      return "var(--success)";
    default:
      return "var(--warning)";
  }
}

export function verdictLabel(v: Verdict | string): string {
  switch (v) {
    case "fake":
      return "FAKE";
    case "real":
      return "REAL";
    case "uncertain":
      return "UNCERTAIN";
    default:
      return v.toUpperCase();
  }
}

export function probColor(p: number): string {
  if (p >= 0.65) return "var(--danger)";
  if (p <= 0.35) return "var(--success)";
  return "var(--warning)";
}

export function clamp01(v: number): number {
  return Math.min(1, Math.max(0, v));
}

export function isValidUrl(s: string): boolean {
  try {
    const u = new URL(s.trim());
    return u.protocol === "http:" || u.protocol === "https:";
  } catch {
    return false;
  }
}
