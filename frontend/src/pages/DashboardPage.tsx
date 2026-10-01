import { useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Activity, BarChart3, BookOpen, Database, Grid2x2, Loader2, MessageSquare, Target } from "lucide-react";
import { api } from "../api";
import { ErrorBanner } from "../components/ErrorBanner";
import { SectionCard } from "../components/SectionCard";
import type { DatasetInfo, ModelMetrics, ModelMetricsResponse, UsageMetrics } from "../types";
import { datasetLabel, fmtMs, fmtNumber, modelLabel, pct } from "../utils";

const SERIES = {
  accuracy: "#8b7cf6",
  f1: "#22d3ee",
  auc: "#f59e0b",
};
const VERDICT_COLORS: Record<string, string> = {
  fake: "#f43f5e",
  real: "#22c55e",
  uncertain: "#f59e0b",
};
const FAKE_COLOR = "#f43f5e";
const REAL_COLOR = "#22c55e";

const tooltipStyle = {
  background: "rgba(15, 19, 32, 0.95)",
  border: "1px solid rgba(255,255,255,0.08)",
  borderRadius: 10,
  color: "#e6e8f0",
  fontSize: 12,
};

function Kpi({ label, value, sub, icon }: { label: string; value: string; sub?: string; icon?: React.ReactNode }) {
  return (
    <div className="kpi">
      <div className="kpi__top">
        <span className="kpi__label">{label}</span>
        {icon && <span className="kpi__icon">{icon}</span>}
      </div>
      <div className="kpi__value">{value}</div>
      {sub && <div className="kpi__sub">{sub}</div>}
    </div>
  );
}

function ConfusionMatrix({ m }: { m: ModelMetrics }) {
  const [[tn, fp], [fn, tp]] = m.confusion.length === 2 ? m.confusion : [[0, 0], [0, 0]];
  const total = tn + fp + fn + tp || 1;
  const cell = (label: string, n: number, kind: "good" | "bad") => (
    <div className={`cm__cell cm__cell--${kind}`} style={{ ["--a" as string]: (0.15 + (n / total) * 0.85).toFixed(2) }}>
      <span className="cm__n">{fmtNumber(n)}</span>
      <span className="cm__l">{label}</span>
      <span className="cm__p">{pct(n / total, 1)}</span>
    </div>
  );
  return (
    <div className="cm">
      <div className="cm__axis cm__axis--top">Predicted</div>
      <div className="cm__axis cm__axis--left">Actual</div>
      <div className="cm__grid">
        <div className="cm__hdr" />
        <div className="cm__hdr">Real</div>
        <div className="cm__hdr">Fake</div>
        <div className="cm__hdr cm__hdr--row">Real</div>
        {cell("True negative", tn, "good")}
        {cell("False positive", fp, "bad")}
        <div className="cm__hdr cm__hdr--row">Fake</div>
        {cell("False negative", fn, "bad")}
        {cell("True positive", tp, "good")}
      </div>
    </div>
  );
}

function datasetEntries(dataset: ModelMetricsResponse["dataset"]): { key: string; info: DatasetInfo }[] {
  return Object.entries(dataset)
    .filter(([k, v]) => k !== "total" && k !== "fake_share" && v && typeof v === "object")
    .map(([key, v]) => ({ key, info: v as DatasetInfo }));
}

export function DashboardPage() {
  const [models, setModels] = useState<ModelMetricsResponse | null>(null);
  const [usage, setUsage] = useState<UsageMetrics | null>(null);
  const [errM, setErrM] = useState<string | null>(null);
  const [errU, setErrU] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    Promise.allSettled([api.modelMetrics(), api.usageMetrics()]).then(([m, u]) => {
      if (cancelled) return;
      if (m.status === "fulfilled") setModels(m.value);
      else setErrM((m.reason as Error).message);
      if (u.status === "fulfilled") setUsage(u.value);
      else setErrU((u.reason as Error).message);
      setLoading(false);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  const modelRows = useMemo(() => {
    if (!models) return [];
    const rows = Object.entries(models.models).map(([k, m]) => ({ key: k, name: modelLabel(k), m }));
    rows.push({ key: "ensemble", name: "Stacked ensemble", m: models.ensemble });
    return rows;
  }, [models]);

  const compareData = modelRows.map((r) => ({
    name: r.name.replace(" + ", "+").replace(" embeddings", ""),
    Accuracy: +(r.m.accuracy * 100).toFixed(1),
    F1: +(r.m.f1 * 100).toFixed(1),
    "ROC-AUC": +(r.m.roc_auc * 100).toFixed(1),
  }));

  const datasets = models ? datasetEntries(models.dataset) : [];
  const compData = datasets.map((d) => ({
    name: datasetLabel(d.key),
    Fake: d.info.fake ?? 0,
    Real: d.info.real ?? 0,
  }));
  const perDatasetKeys = useMemo(() => {
    const keys = new Set<string>();
    modelRows.forEach((r) => Object.keys(r.m.per_dataset ?? {}).forEach((k) => keys.add(k)));
    return Array.from(keys);
  }, [modelRows]);

  const verdictData = usage
    ? Object.entries(usage.verdicts).map(([name, value]) => ({ name, value }))
    : [];

  return (
    <div className="page">
      <header className="page__head">
        <div>
          <h1 className="page__title">
            <BarChart3 size={24} /> Dashboard
          </h1>
          <p className="page__sub">Offline evaluation of the ensemble on held-out test splits, plus live usage of this deployment.</p>
        </div>
      </header>

      {loading && (
        <div className="empty">
          <Loader2 className="spin" size={22} /> Loading metrics…
        </div>
      )}
      {errM && <ErrorBanner message={`Model metrics unavailable: ${errM}`} />}
      {errU && <ErrorBanner message={`Usage metrics unavailable: ${errU}`} kind="warning" />}

      {models && (
        <>
          <div className="kpis">
            <Kpi label="Ensemble accuracy" value={pct(models.ensemble.accuracy)} sub="held-out test" icon={<Target size={16} />} />
            <Kpi label="Ensemble F1" value={pct(models.ensemble.f1)} sub={`precision ${pct(models.ensemble.precision, 0)} · recall ${pct(models.ensemble.recall, 0)}`} />
            <Kpi label="ROC-AUC" value={pct(models.ensemble.roc_auc)} sub="ranking quality" />
            <Kpi label="Test size" value={fmtNumber(models.ensemble.n)} sub={`${fmtNumber(models.dataset.total)} total rows · ${pct(models.dataset.fake_share, 0)} fake`} icon={<Database size={16} />} />
            <Kpi label="Analyses run" value={usage ? fmtNumber(usage.analyses) : "—"} sub={usage ? `avg ${fmtMs(usage.avg_latency_ms)}` : "usage unavailable"} icon={<Activity size={16} />} />
          </div>

          <div className="grid grid--2">
            <SectionCard title="Model comparison" subtitle="Accuracy, F1 and ROC-AUC on the combined test split (%)" icon={<BarChart3 size={18} />}>
              <div className="chart">
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={compareData} margin={{ top: 8, right: 8, left: -16, bottom: 8 }}>
                    <CartesianGrid stroke="rgba(255,255,255,0.06)" vertical={false} />
                    <XAxis dataKey="name" tick={{ fill: "#9aa3b8", fontSize: 11 }} interval={0} />
                    <YAxis domain={[50, 100]} tick={{ fill: "#9aa3b8", fontSize: 11 }} />
                    <Tooltip contentStyle={tooltipStyle} cursor={{ fill: "rgba(255,255,255,0.04)" }} />
                    <Legend wrapperStyle={{ fontSize: 12 }} />
                    <Bar dataKey="Accuracy" fill={SERIES.accuracy} radius={[6, 6, 0, 0]} />
                    <Bar dataKey="F1" fill={SERIES.f1} radius={[6, 6, 0, 0]} />
                    <Bar dataKey="ROC-AUC" fill={SERIES.auc} radius={[6, 6, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              {Object.keys(models.meta_weights).length > 0 && (
                <div className="weights">
                  <span className="muted small">Meta-learner weights:</span>
                  {Object.entries(models.meta_weights).map(([k, w]) => (
                    <span key={k} className="chip">
                      {modelLabel(k)} <b>{w.toFixed(2)}</b>
                    </span>
                  ))}
                </div>
              )}
            </SectionCard>

            <SectionCard title="Ensemble confusion matrix" subtitle={`n = ${fmtNumber(models.ensemble.n)} · positive class = fake`} icon={<Grid2x2 size={18} />}>
              <ConfusionMatrix m={models.ensemble} />
            </SectionCard>
          </div>

          <SectionCard title="Per-dataset performance" subtitle="How each model generalises across sources" icon={<Database size={18} />}>
            <div className="table-wrap">
              <table className="table">
                <thead>
                  <tr>
                    <th>Model</th>
                    {perDatasetKeys.map((k) => (
                      <th key={k} colSpan={3} className="table__group">
                        {datasetLabel(k)}
                      </th>
                    ))}
                  </tr>
                  <tr className="table__sub">
                    <th />
                    {perDatasetKeys.map((k) => (
                      <SubHeads key={k} />
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {modelRows.map((r) => (
                    <tr key={r.key} className={r.key === "ensemble" ? "table__row--hl" : ""}>
                      <td>{r.name}</td>
                      {perDatasetKeys.map((k) => {
                        const d = r.m.per_dataset?.[k];
                        return d ? (
                          <PerDatasetCells key={k} acc={d.accuracy} f1={d.f1} auc={d.roc_auc} n={d.n} />
                        ) : (
                          <PerDatasetCells key={k} acc={null} f1={null} auc={null} n={null} />
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="hint">{models.protocol}</p>
          </SectionCard>

          <div className="grid grid--2">
            <SectionCard title="Dataset composition" subtitle="Rows per dataset, fake vs real" icon={<Database size={18} />}>
              <div className="chart">
                <ResponsiveContainer width="100%" height={260}>
                  <BarChart data={compData} layout="vertical" margin={{ top: 8, right: 16, left: 8, bottom: 8 }}>
                    <CartesianGrid stroke="rgba(255,255,255,0.06)" horizontal={false} />
                    <XAxis type="number" tick={{ fill: "#9aa3b8", fontSize: 11 }} />
                    <YAxis type="category" dataKey="name" tick={{ fill: "#9aa3b8", fontSize: 12 }} width={80} />
                    <Tooltip contentStyle={tooltipStyle} cursor={{ fill: "rgba(255,255,255,0.04)" }} />
                    <Legend wrapperStyle={{ fontSize: 12 }} />
                    <Bar dataKey="Real" stackId="a" fill={REAL_COLOR} radius={[0, 0, 0, 0]} />
                    <Bar dataKey="Fake" stackId="a" fill={FAKE_COLOR} radius={[0, 6, 6, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <div className="splits">
                {datasets.map((d) => (
                  <div key={d.key} className="splits__row">
                    <b>{datasetLabel(d.key)}</b>
                    <span className="muted small">
                      {fmtNumber(d.info.rows)} rows
                      {d.info.splits && ` · train ${fmtNumber(d.info.splits.train)} / valid ${fmtNumber(d.info.splits.valid)} / test ${fmtNumber(d.info.splits.test)}`}
                    </span>
                  </div>
                ))}
              </div>
            </SectionCard>

            <SectionCard title="About the datasets" subtitle="Where the training signal comes from" icon={<BookOpen size={18} />}>
              <div className="notes-cards">
                <div className="note">
                  <h4>LIAR</h4>
                  <p>
                    12.8k short political statements from PolitiFact, each with a six-way truthfulness label (pants-on-fire → true), speaker metadata and context. We binarise labels into fake / real. <em>Wang, 2017 — "Liar, Liar Pants on Fire".</em>
                  </p>
                </div>
                <div className="note">
                  <h4>FakeNewsNet</h4>
                  <p>
                    Full news articles fact-checked by PolitiFact (politics) and GossipCop (entertainment), with social context. GossipCop is the largest source and the most style-driven. <em>Shu et al., 2018.</em>
                  </p>
                </div>
                <div className="note note--muted">
                  <p>
                    Splits are stratified by dataset and label; all reported metrics are on the held-out test split the models never saw during training or meta-learner fitting.
                  </p>
                </div>
              </div>
            </SectionCard>
          </div>
        </>
      )}

      {usage && (
        <>
          <h2 className="section-title">Live usage</h2>
          <div className="grid grid--3">
            <SectionCard title="Verdict distribution" subtitle={`${fmtNumber(usage.analyses)} analyses`} icon={<Target size={18} />}>
              {verdictData.length === 0 ? (
                <div className="empty empty--sm">No analyses yet.</div>
              ) : (
                <div className="chart">
                  <ResponsiveContainer width="100%" height={220}>
                    <PieChart>
                      <Pie data={verdictData} dataKey="value" nameKey="name" innerRadius={55} outerRadius={85} paddingAngle={3} stroke="none">
                        {verdictData.map((d) => (
                          <Cell key={d.name} fill={VERDICT_COLORS[d.name] ?? "#8b7cf6"} />
                        ))}
                      </Pie>
                      <Tooltip contentStyle={tooltipStyle} />
                      <Legend wrapperStyle={{ fontSize: 12 }} />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              )}
            </SectionCard>

            <SectionCard title="Analyses by day" subtitle="Volume over time" icon={<Activity size={18} />}>
              {usage.by_day.length === 0 ? (
                <div className="empty empty--sm">No data yet.</div>
              ) : (
                <div className="chart">
                  <ResponsiveContainer width="100%" height={220}>
                    <LineChart data={usage.by_day} margin={{ top: 8, right: 8, left: -20, bottom: 0 }}>
                      <CartesianGrid stroke="rgba(255,255,255,0.06)" vertical={false} />
                      <XAxis dataKey="day" tick={{ fill: "#9aa3b8", fontSize: 10 }} tickFormatter={(d: string) => d.slice(5)} />
                      <YAxis allowDecimals={false} tick={{ fill: "#9aa3b8", fontSize: 11 }} />
                      <Tooltip contentStyle={tooltipStyle} />
                      <Line type="monotone" dataKey="count" stroke={SERIES.accuracy} strokeWidth={2.5} dot={{ r: 3, fill: SERIES.accuracy }} activeDot={{ r: 5 }} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )}
            </SectionCard>

            <SectionCard title="Human feedback" subtitle="Agreement with user ratings" icon={<MessageSquare size={18} />}>
              <div className="agree">
                <div className="agree__big">{usage.feedback.agreement === null ? "—" : pct(usage.feedback.agreement, 0)}</div>
                <div className="agree__sub">agreement rate</div>
                <div className="meter meter--lg">
                  <span style={{ width: `${Math.round((usage.feedback.agreement ?? 0) * 100)}%`, background: "var(--success)" }} />
                </div>
                <div className="agree__row">
                  <span>
                    <b style={{ color: "var(--success)" }}>{usage.feedback.correct}</b> correct
                  </span>
                  <span>
                    <b style={{ color: "var(--danger)" }}>{usage.feedback.incorrect}</b> incorrect
                  </span>
                  <span>
                    <b>{usage.feedback.total}</b> total
                  </span>
                </div>
              </div>
            </SectionCard>
          </div>
        </>
      )}
    </div>
  );
}

function SubHeads() {
  return (
    <>
      <th>Acc</th>
      <th>F1</th>
      <th>AUC</th>
    </>
  );
}

function PerDatasetCells({ acc, f1, auc, n }: { acc: number | null; f1: number | null; auc: number | null; n: number | null }) {
  return (
    <>
      <td title={n !== null ? `n = ${fmtNumber(n)}` : undefined}>{pct(acc)}</td>
      <td>{pct(f1)}</td>
      <td>{auc === null ? "—" : pct(auc)}</td>
    </>
  );
}
