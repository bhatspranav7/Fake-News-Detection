import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Activity, FileText, Link2, Loader2, Rocket, ScanSearch, Sparkles, Square, Zap } from "lucide-react";
import { analyzeStream, api } from "../api";
import { AgentTimeline, type TimelineStep } from "../components/AgentTimeline";
import { ErrorBanner } from "../components/ErrorBanner";
import { ResultView } from "../components/ResultView";
import { useHealthContext } from "../hooks/HealthContext";
import type { AnalysisResult, Example, Mode, StepEvent } from "../types";
import { AGENT_ORDER, isValidUrl } from "../utils";

const FALLBACK_EXAMPLES: Example[] = [
  {
    label: "Miracle cure",
    text: "BREAKING: Scientists CONFIRM that drinking lemon water every morning cures cancer in 30 days — doctors don't want you to know this!",
  },
  {
    label: "Policy report",
    text: "The Federal Reserve held interest rates steady on Wednesday, citing cooling inflation and a resilient labor market, according to its post-meeting statement.",
  },
  {
    label: "Celebrity hoax",
    text: "Shocking: famous actor secretly arrested at airport after customs found 'illegal alien technology' in his luggage, insiders claim.",
  },
];

const MIN_TEXT = 15;

function initialSteps(mode: Mode): TimelineStep[] {
  const order = mode === "fast" ? ["ingest", "classifier", "judge"] : AGENT_ORDER;
  return order.map((agent) => ({ agent, status: "pending", summary: "", elapsed_ms: null }));
}

export function AnalyzePage() {
  const { health, online } = useHealthContext();
  const [inputKind, setInputKind] = useState<"text" | "url">("text");
  const [text, setText] = useState("");
  const [url, setUrl] = useState("");
  const [mode, setMode] = useState<Mode>("deep");
  const [examples, setExamples] = useState<Example[]>(FALLBACK_EXAMPLES);
  const [running, setRunning] = useState(false);
  const [steps, setSteps] = useState<TimelineStep[]>([]);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [startedAt, setStartedAt] = useState<number | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const abortRef = useRef<AbortController | null>(null);
  const resultRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    let cancelled = false;
    api
      .examples()
      .then((r) => {
        if (!cancelled && r.examples && r.examples.length > 0) setExamples(r.examples.slice(0, 6));
      })
      .catch(() => {
        /* keep fallback */
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!running || startedAt === null) return;
    const id = window.setInterval(() => setElapsed(Date.now() - startedAt), 100);
    return () => window.clearInterval(id);
  }, [running, startedAt]);

  useEffect(() => () => abortRef.current?.abort(), []);

  const validation = useMemo(() => {
    if (inputKind === "text") {
      if (text.trim().length === 0) return "Paste a headline or article to analyze.";
      if (text.trim().length < MIN_TEXT) return `Add a little more text (at least ${MIN_TEXT} characters).`;
      return null;
    }
    if (url.trim().length === 0) return "Enter an article URL.";
    if (!isValidUrl(url)) return "That doesn't look like a valid http(s) URL.";
    return null;
  }, [inputKind, text, url]);

  const applyStep = useCallback((ev: StepEvent) => {
    setSteps((prev) => {
      const idx = prev.findIndex((s) => s.agent === ev.agent);
      const next: TimelineStep = { agent: ev.agent, status: ev.status, summary: ev.summary, elapsed_ms: ev.elapsed_ms };
      if (idx === -1) return [...prev, next];
      const copy = prev.slice();
      copy[idx] = next;
      return copy;
    });
  }, []);

  const run = useCallback(async () => {
    if (validation || running) return;
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setError(null);
    setResult(null);
    setSteps(initialSteps(mode));
    setRunning(true);
    setStartedAt(Date.now());
    setElapsed(0);

    const body = inputKind === "text" ? { text: text.trim(), mode } : { url: url.trim(), mode };

    await analyzeStream(
      body,
      {
        onStep: applyStep,
        onResult: (r) => {
          setResult(r);
          // Mark any still-pending steps from the final agent list.
          setSteps((prev) =>
            prev.map((s) => {
              const a = r.agents.find((x) => x.name === s.agent);
              if (!a) return s.status === "pending" ? { ...s, status: "skipped" } : s;
              return { agent: a.name, status: a.status, summary: a.summary, elapsed_ms: a.elapsed_ms };
            }),
          );
          window.setTimeout(() => resultRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 50);
        },
        onError: (msg) => setError(msg),
      },
      controller.signal,
    );

    if (!controller.signal.aborted) setRunning(false);
  }, [applyStep, inputKind, mode, running, text, url, validation]);

  const cancel = () => {
    abortRef.current?.abort();
    setRunning(false);
    setSteps((prev) => prev.map((s) => (s.status === "running" || s.status === "pending" ? { ...s, status: "skipped" } : s)));
  };

  const useExample = (ex: Example) => {
    setInputKind("text");
    setText(ex.text);
    setResult(null);
    setError(null);
  };

  const llmOffline = online && health !== null && !health.llm?.ok;

  return (
    <div className="page">
      <header className="hero">
        <div className="hero__badge">
          <Sparkles size={14} /> Hackathon build · agentic pipeline
        </div>
        <h1 className="hero__title">
          Veri<span className="grad">Fact</span>
        </h1>
        <p className="hero__tagline">Agentic fake-news detection — deep learning ensemble, LLM reasoning, live evidence</p>
      </header>

      {llmOffline && (
        <ErrorBanner kind="warning" message="LLM offline — deep mode falls back to ML only (no claim extraction or LLM reasoning)." />
      )}
      {!online && health === null && (
        <ErrorBanner kind="info" message="Backend not reachable yet. Start the API and this page will reconnect automatically." />
      )}

      <section className="card input-card">
        <div className="input-card__top">
          <div className="seg" role="tablist" aria-label="Input type">
            <button role="tab" aria-selected={inputKind === "text"} className={`seg__btn ${inputKind === "text" ? "seg__btn--on" : ""}`} onClick={() => setInputKind("text")}>
              <FileText size={15} /> Text
            </button>
            <button role="tab" aria-selected={inputKind === "url"} className={`seg__btn ${inputKind === "url" ? "seg__btn--on" : ""}`} onClick={() => setInputKind("url")}>
              <Link2 size={15} /> URL
            </button>
          </div>
          <span className="muted small">{inputKind === "text" ? `${text.trim().split(/\s+/).filter(Boolean).length} words` : "We fetch and extract the article"}</span>
        </div>

        {inputKind === "text" ? (
          <textarea
            className="input input--lg"
            placeholder="Paste a headline, tweet, or full article…"
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={7}
            onKeyDown={(e) => {
              if ((e.ctrlKey || e.metaKey) && e.key === "Enter") void run();
            }}
          />
        ) : (
          <input
            className="input input--lg"
            type="url"
            placeholder="https://example.com/news/some-article"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") void run();
            }}
          />
        )}

        <div className="modes" role="radiogroup" aria-label="Analysis mode">
          <button role="radio" aria-checked={mode === "fast"} className={`mode ${mode === "fast" ? "mode--on" : ""}`} onClick={() => setMode("fast")}>
            <Zap size={18} />
            <span className="mode__text">
              <span className="mode__name">Fast</span>
              <span className="mode__desc">ML ensemble only · ~1 s</span>
            </span>
          </button>
          <button role="radio" aria-checked={mode === "deep"} className={`mode ${mode === "deep" ? "mode--on" : ""}`} onClick={() => setMode("deep")}>
            <Rocket size={18} />
            <span className="mode__text">
              <span className="mode__name">Deep</span>
              <span className="mode__desc">ML + LLM agents + web evidence · 10–30 s</span>
            </span>
          </button>
        </div>

        <div className="input-card__actions">
          {running ? (
            <button className="btn btn--danger" onClick={cancel}>
              <Square size={16} /> Stop
            </button>
          ) : (
            <button className="btn btn--primary btn--lg" onClick={() => void run()} disabled={validation !== null} title={validation ?? "Analyze (Ctrl+Enter)"}>
              <ScanSearch size={18} /> Analyze
            </button>
          )}
          {validation && !running && <span className="muted small">{validation}</span>}
        </div>

        <div className="examples">
          <span className="examples__label">Try an example:</span>
          <div className="examples__chips">
            {examples.map((ex) => (
              <button key={ex.label} className="chip chip--btn" onClick={() => useExample(ex)} disabled={running}>
                {ex.label}
              </button>
            ))}
          </div>
        </div>
      </section>

      {error && <ErrorBanner message={error} onClose={() => setError(null)} />}

      {(running || (steps.length > 0 && !result)) && (
        <section className="card live">
          <header className="card__head">
            <div className="card__title">
              <span className="card__icon">{running ? <Loader2 size={18} className="spin" /> : <Activity size={18} />}</span>
              <div>
                <h3>{running ? "Agents working…" : "Run finished"}</h3>
                <p className="card__subtitle">
                  {mode === "deep" ? "Seven-agent pipeline orchestrated with LangGraph" : "Fast path — ensemble classifier only"}
                </p>
              </div>
            </div>
            <span className="live__clock">{(elapsed / 1000).toFixed(1)} s</span>
          </header>
          <div className="card__body">
            <AgentTimeline steps={steps} />
          </div>
        </section>
      )}

      {result && (
        <div ref={resultRef}>
          {steps.length > 0 && (
            <details className="card live live--collapsed" open={false}>
              <summary className="live__summary">
                <Activity size={16} /> Agent timeline · {steps.filter((s) => s.status === "done").length}/{steps.length} completed
              </summary>
              <div className="card__body">
                <AgentTimeline steps={steps} compact />
              </div>
            </details>
          )}
          <ResultView result={result} showAgents={false} />
        </div>
      )}
    </div>
  );
}
