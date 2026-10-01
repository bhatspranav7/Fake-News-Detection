export type Mode = "fast" | "deep";
export type Verdict = "fake" | "real" | "uncertain";
export type ClaimVerdict = "supported" | "refuted" | "unverified";
export type Stance = "supports" | "refutes" | "neutral";
export type Credibility = "high" | "medium" | "low" | "unknown";
export type AgentStatus = "running" | "done" | "error" | "skipped";

export interface Health {
  status: string;
  version: string;
  models_loaded: string[];
  llm: { provider: string; model: string; ok: boolean };
  index: { analyses: number };
}

export interface AnalyzeRequest {
  text?: string;
  url?: string;
  mode?: Mode;
}

export interface StepEvent {
  agent: string;
  status: "running" | "done" | "error";
  summary: string;
  elapsed_ms: number | null;
  payload?: unknown;
}

export interface Evidence {
  title: string;
  url: string;
  snippet: string;
  stance: Stance;
}

export interface Claim {
  claim: string;
  verdict: ClaimVerdict;
  confidence: number;
  evidence: Evidence[];
}

export interface TopToken {
  token: string;
  weight: number;
}

export interface AgentRun {
  name: string;
  status: "done" | "error" | "skipped";
  elapsed_ms: number;
  summary: string;
}

export interface AnalysisResult {
  id: string;
  created_at: string;
  mode: Mode;
  input: {
    text: string;
    url: string | null;
    title: string | null;
    domain: string | null;
    word_count: number;
  };
  verdict: Verdict;
  fake_probability: number;
  confidence: number;
  ml: {
    ensemble_prob: number;
    models: Record<string, number>;
    top_tokens: TopToken[];
    style_flags: string[];
  };
  claims: Claim[];
  source: {
    domain: string | null;
    credibility: Credibility;
    score: number | null;
    notes: string[];
  };
  llm: {
    provider: string;
    model: string;
    summary: string;
    red_flags: string[];
    reasoning: string;
  } | null;
  agents: AgentRun[];
  latency_ms: number;
}

export interface FeedbackRequest {
  analysis_id: string;
  rating: "correct" | "incorrect";
  comment?: string;
}

export interface HistoryItem {
  id: string;
  created_at: string;
  snippet: string;
  verdict: string;
  fake_probability: number;
  confidence: number;
  mode: string;
  domain: string | null;
}

export interface HistoryResponse {
  items: HistoryItem[];
}

export interface Example {
  label: string;
  text: string;
}

export interface ExamplesResponse {
  examples: Example[];
}

export interface PerDatasetMetrics {
  accuracy: number;
  f1: number;
  roc_auc: number | null;
  n: number;
}

export interface ModelMetrics {
  model: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  roc_auc: number;
  confusion: number[][];
  n: number;
  per_dataset: Record<string, PerDatasetMetrics>;
  train_seconds?: number;
}

export interface DatasetInfo {
  rows?: number;
  fake?: number;
  real?: number;
  splits?: { train?: number; valid?: number; test?: number };
}

export interface ModelMetricsResponse {
  dataset: { total: number; fake_share: number; [datasetName: string]: unknown };
  models: Record<string, ModelMetrics>;
  ensemble: ModelMetrics;
  meta_weights: Record<string, number>;
  protocol: string;
}

export interface UsageMetrics {
  analyses: number;
  verdicts: Record<string, number>;
  avg_latency_ms: number;
  feedback: {
    total: number;
    correct: number;
    incorrect: number;
    agreement: number | null;
  };
  by_day: { day: string; count: number }[];
}
