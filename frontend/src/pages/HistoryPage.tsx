import { useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { ArrowLeft, Clock, Globe, History, Loader2, RefreshCw } from "lucide-react";
import { api } from "../api";
import { ErrorBanner } from "../components/ErrorBanner";
import { ResultView } from "../components/ResultView";
import type { AnalysisResult, HistoryItem } from "../types";
import { fmtDate, pct, truncate, verdictColor, verdictLabel } from "../utils";

export function HistoryPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [items, setItems] = useState<HistoryItem[]>([]);
  const [loadingList, setLoadingList] = useState(true);
  const [listError, setListError] = useState<string | null>(null);

  const [detail, setDetail] = useState<AnalysisResult | null>(null);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [detailError, setDetailError] = useState<string | null>(null);

  const loadList = () => {
    setLoadingList(true);
    setListError(null);
    api
      .history(30)
      .then((r) => setItems(r.items))
      .catch((e: Error) => setListError(e.message))
      .finally(() => setLoadingList(false));
  };

  useEffect(loadList, []);

  useEffect(() => {
    if (!id) {
      setDetail(null);
      setDetailError(null);
      return;
    }
    let cancelled = false;
    setLoadingDetail(true);
    setDetailError(null);
    api
      .analysis(id)
      .then((r) => {
        if (!cancelled) setDetail(r);
      })
      .catch((e: Error) => {
        if (!cancelled) setDetailError(e.message);
      })
      .finally(() => {
        if (!cancelled) setLoadingDetail(false);
      });
    return () => {
      cancelled = true;
    };
  }, [id]);

  if (id) {
    return (
      <div className="page">
        <div className="page__bar">
          <button className="btn btn--ghost" onClick={() => navigate("/history")}>
            <ArrowLeft size={16} /> Back to history
          </button>
          <span className="muted small mono">{id}</span>
        </div>
        {loadingDetail && (
          <div className="empty">
            <Loader2 className="spin" size={22} /> Loading analysis…
          </div>
        )}
        {detailError && <ErrorBanner message={detailError} />}
        {detail && <ResultView result={detail} />}
      </div>
    );
  }

  return (
    <div className="page">
      <header className="page__head">
        <div>
          <h1 className="page__title">
            <History size={24} /> History
          </h1>
          <p className="page__sub">Recent analyses stored by the backend. Click one to reopen the full report.</p>
        </div>
        <button className="btn btn--ghost" onClick={loadList} disabled={loadingList}>
          <RefreshCw size={16} className={loadingList ? "spin" : ""} /> Refresh
        </button>
      </header>

      {listError && <ErrorBanner message={listError} />}

      {loadingList && items.length === 0 && (
        <div className="empty">
          <Loader2 className="spin" size={22} /> Loading…
        </div>
      )}

      {!loadingList && !listError && items.length === 0 && (
        <div className="empty">
          No analyses yet. <Link to="/">Run your first one →</Link>
        </div>
      )}

      {items.length > 0 && (
        <ul className="hist">
          {items.map((it, i) => (
            <li key={it.id} style={{ animationDelay: `${i * 30}ms` }}>
              <Link to={`/history/${encodeURIComponent(it.id)}`} className="hist__item">
                <span className="hist__verdict" style={{ ["--c" as string]: verdictColor(it.verdict) }}>
                  {verdictLabel(it.verdict)}
                </span>
                <span className="hist__main">
                  <span className="hist__snippet">{truncate(it.snippet, 160)}</span>
                  <span className="hist__meta">
                    <span>
                      <Clock size={12} /> {fmtDate(it.created_at)}
                    </span>
                    {it.domain && (
                      <span>
                        <Globe size={12} /> {it.domain}
                      </span>
                    )}
                    <span className={`chip chip--${it.mode}`}>{it.mode}</span>
                  </span>
                </span>
                <span className="hist__nums">
                  <span className="hist__num">
                    <small>fake prob</small>
                    <b style={{ color: verdictColor(it.verdict) }}>{pct(it.fake_probability, 0)}</b>
                  </span>
                  <span className="hist__num">
                    <small>confidence</small>
                    <b>{pct(it.confidence, 0)}</b>
                  </span>
                </span>
              </Link>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
