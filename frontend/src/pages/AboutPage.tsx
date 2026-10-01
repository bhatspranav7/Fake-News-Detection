import { BrainCircuit, Cpu, Database, GitBranch, Globe, Info, Layers, Plug, Scale, Search, ShieldCheck, Wrench } from "lucide-react";
import type { ReactNode } from "react";
import { SectionCard } from "../components/SectionCard";

interface Stage {
  name: string;
  kind: "input" | "agent" | "ml" | "llm" | "web" | "judge";
  icon: ReactNode;
  desc: string;
  bullets?: string[];
}

const STAGES: Stage[] = [
  { name: "Input", kind: "input", icon: <Globe size={18} />, desc: "Headline, article text, or a URL." },
  {
    name: "Ingest agent",
    kind: "agent",
    icon: <Wrench size={18} />,
    desc: "Fetches the page, extracts the article body & title, resolves the domain, normalises text.",
  },
  {
    name: "Classifier",
    kind: "ml",
    icon: <Layers size={18} />,
    desc: "Deep learning ensemble produces a calibrated fake probability plus token-level attributions.",
    bullets: ["TF-IDF + Logistic Regression", "MiniLM embeddings + XGBoost", "DistilBERT fine-tuned → ONNX int8", "Stacked meta-learner"],
  },
  { name: "Claim extractor", kind: "llm", icon: <BrainCircuit size={18} />, desc: "LLM pulls out the atomic, checkable claims from the text." },
  {
    name: "Evidence retriever",
    kind: "web",
    icon: <Search size={18} />,
    desc: "Searches DuckDuckGo and Wikipedia for each claim and collects snippets.",
  },
  { name: "Fact checker", kind: "llm", icon: <ShieldCheck size={18} />, desc: "LLM reads the evidence and labels each claim supported / refuted / unverified." },
  { name: "Source credibility", kind: "agent", icon: <Scale size={18} />, desc: "Scores the publishing domain against a curated credibility list and heuristics." },
  {
    name: "Judge",
    kind: "judge",
    icon: <Cpu size={18} />,
    desc: "Fuses ML probability, claim verdicts, source score and LLM red flags into the final verdict & confidence.",
  },
];

const TECH = [
  "Python 3.11",
  "FastAPI",
  "LangGraph",
  "PyTorch",
  "Transformers",
  "DistilBERT",
  "ONNX Runtime (int8)",
  "scikit-learn",
  "XGBoost",
  "sentence-transformers / MiniLM",
  "DuckDuckGo Search",
  "Wikipedia API",
  "MCP server",
  "SSE streaming",
  "React 18",
  "Vite",
  "TypeScript",
  "Recharts",
];

export function AboutPage() {
  return (
    <div className="page">
      <header className="page__head">
        <div>
          <h1 className="page__title">
            <Info size={24} /> How VeriFact works
          </h1>
          <p className="page__sub">
            A supervised deep-learning ensemble gives a fast, calibrated signal; a graph of LLM agents then gathers live evidence and reasons about it. The Judge fuses both.
          </p>
        </div>
      </header>

      <SectionCard title="Pipeline" subtitle="Each node is a LangGraph agent; results stream to the UI as they finish" icon={<GitBranch size={18} />}>
        <ol className="pipe">
          {STAGES.map((s, i) => (
            <li key={s.name} className={`pipe__stage pipe__stage--${s.kind}`} style={{ animationDelay: `${i * 70}ms` }}>
              <div className="pipe__node">
                <span className="pipe__icon">{s.icon}</span>
                <span className="pipe__name">{s.name}</span>
              </div>
              <p className="pipe__desc">{s.desc}</p>
              {s.bullets && (
                <ul className="pipe__bullets">
                  {s.bullets.map((b) => (
                    <li key={b}>{b}</li>
                  ))}
                </ul>
              )}
              {i < STAGES.length - 1 && (
                <span className="pipe__arrow" aria-hidden="true">
                  <svg viewBox="0 0 24 24" width="22" height="22">
                    <path d="M4 12h14M13 6l6 6-6 6" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                </span>
              )}
            </li>
          ))}
        </ol>
        <div className="legend">
          <span>
            <i className="legend-dot" style={{ background: "var(--accent)" }} /> ML
          </span>
          <span>
            <i className="legend-dot" style={{ background: "var(--cyan)" }} /> LLM agent
          </span>
          <span>
            <i className="legend-dot" style={{ background: "var(--warning)" }} /> Web retrieval
          </span>
          <span>
            <i className="legend-dot" style={{ background: "var(--success)" }} /> Fusion
          </span>
        </div>
      </SectionCard>

      <div className="grid grid--2">
        <SectionCard title="Deep learning ensemble" subtitle="Three diverse base models, one meta-learner" icon={<Layers size={18} />}>
          <ul className="list">
            <li>
              <b>TF-IDF + Logistic Regression</b> — word & character n-grams; fast, interpretable, and the source of the signed token attributions shown in the “Why” card.
            </li>
            <li>
              <b>MiniLM embeddings + XGBoost</b> — sentence-transformer embeddings concatenated with hand-crafted style features (caps ratio, exclamation density, clickbait cues) fed to gradient-boosted trees.
            </li>
            <li>
              <b>DistilBERT (fine-tuned)</b> — a transformer fine-tuned end-to-end on the combined corpus, exported to ONNX and quantised to int8 for sub-second CPU inference.
            </li>
            <li>
              <b>Stacked meta-learner</b> — a logistic regression trained on out-of-fold base-model probabilities, giving calibrated fused output and per-model weights.
            </li>
          </ul>
        </SectionCard>

        <SectionCard title="Agentic layer" subtitle="LangGraph orchestration, streamed over SSE" icon={<BrainCircuit size={18} />}>
          <ul className="list">
            <li>
              <b>LangGraph</b> defines the agents as a state graph with conditional edges: fast mode short-circuits straight to the Judge; deep mode fans out claim checks in parallel.
            </li>
            <li>
              <b>Evidence</b> comes from DuckDuckGo web search and the Wikipedia API, with snippets stance-labelled by the fact-checker LLM.
            </li>
            <li>
              <b>Judge</b> combines ML probability, claim verdicts, source credibility and LLM red flags, with explicit uncertainty when signals disagree.
            </li>
            <li>
              <b>Graceful degradation</b> — if the LLM provider is unavailable the pipeline falls back to the ML-only path and says so.
            </li>
          </ul>
        </SectionCard>
      </div>

      <div className="grid grid--2">
        <SectionCard title="Integrations" subtitle="Backend surface area" icon={<Plug size={18} />}>
          <ul className="list">
            <li>
              <b>FastAPI backend</b> — <code>/analyze</code>, streaming <code>/analyze/stream</code>, history, feedback and metrics endpoints.
            </li>
            <li>
              <b>MCP server</b> — exposes <code>verify_claim</code> and <code>analyze_article</code> tools so Claude (or any MCP client) can call VeriFact directly from a conversation.
            </li>
            <li>
              <b>Feedback loop</b> — thumbs up / down ratings are stored alongside each analysis and surfaced as an agreement rate on the dashboard.
            </li>
          </ul>
        </SectionCard>

        <SectionCard title="Datasets" subtitle="Public benchmarks for fake-news detection" icon={<Database size={18} />}>
          <ul className="list">
            <li>
              <b>LIAR</b> (Wang, 2017) — 12.8k manually labelled PolitiFact statements, six truthfulness classes collapsed to binary.
            </li>
            <li>
              <b>FakeNewsNet</b> (Shu et al., 2018) — PolitiFact and GossipCop articles with ground-truth labels from professional fact-checkers.
            </li>
            <li>Stratified train / valid / test splits per dataset; metrics on the dashboard are from the untouched test split.</li>
          </ul>
        </SectionCard>
      </div>

      <SectionCard title="Tech stack" icon={<Cpu size={18} />}>
        <div className="badges">
          {TECH.map((t) => (
            <span key={t} className="chip chip--tech">
              {t}
            </span>
          ))}
        </div>
      </SectionCard>
    </div>
  );
}
