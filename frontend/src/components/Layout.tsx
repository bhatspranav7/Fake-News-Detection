import { useEffect, useState } from "react";
import { NavLink, Outlet, useLocation } from "react-router-dom";
import { BarChart3, History, Info, Menu, ScanSearch, ShieldCheck, X } from "lucide-react";
import { useHealthContext } from "../hooks/HealthContext";

const NAV = [
  { to: "/", label: "Analyze", icon: ScanSearch, end: true },
  { to: "/dashboard", label: "Dashboard", icon: BarChart3, end: false },
  { to: "/history", label: "History", icon: History, end: false },
  { to: "/about", label: "About", icon: Info, end: false },
];

function StatusPill() {
  const { health, online, checking } = useHealthContext();
  if (checking && !health) {
    return (
      <div className="status-pill" title="Checking backend status">
        <span className="status-dot status-dot--pending" />
        <span>Checking API…</span>
      </div>
    );
  }
  if (!online || !health) {
    return (
      <div className="status-pill status-pill--off" title="Backend unreachable">
        <span className="status-dot status-dot--off" />
        <span>API offline</span>
      </div>
    );
  }
  const llmText = health.llm.ok ? health.llm.model : "LLM offline";
  return (
    <div
      className="status-pill status-pill--on"
      title={`v${health.version} · ${health.models_loaded.length} models · ${health.index.analyses} analyses indexed`}
    >
      <span className="status-dot status-dot--on" />
      <span className="status-pill__text">
        API online · <span className={health.llm.ok ? "" : "muted"}>{llmText}</span>
      </span>
    </div>
  );
}

export function Layout() {
  const [open, setOpen] = useState(false);
  const location = useLocation();

  useEffect(() => {
    setOpen(false);
  }, [location.pathname]);

  return (
    <div className="shell">
      <aside className={`sidebar ${open ? "sidebar--open" : ""}`}>
        <div className="sidebar__top">
          <NavLink to="/" className="brand" aria-label="VeriFact home">
            <span className="brand__logo">
              <ShieldCheck size={20} strokeWidth={2.4} />
            </span>
            <span className="brand__text">
              <span className="brand__name">VeriFact</span>
              <span className="brand__sub">Agentic fact-checking</span>
            </span>
          </NavLink>
          <button
            className="icon-btn sidebar__toggle"
            aria-label={open ? "Close navigation" : "Open navigation"}
            aria-expanded={open}
            onClick={() => setOpen((v) => !v)}
          >
            {open ? <X size={20} /> : <Menu size={20} />}
          </button>
        </div>

        <nav className="nav">
          {NAV.map(({ to, label, icon: Icon, end }) => (
            <NavLink
              key={to}
              to={to}
              end={end}
              className={({ isActive }) => `nav__link ${isActive ? "nav__link--active" : ""}`}
            >
              <Icon size={18} />
              <span>{label}</span>
            </NavLink>
          ))}
        </nav>

        <div className="sidebar__bottom">
          <StatusPill />
          <p className="sidebar__foot">Deep learning ensemble · LLM agents · live evidence</p>
        </div>
      </aside>

      <main className="content">
        <Outlet />
      </main>
    </div>
  );
}
