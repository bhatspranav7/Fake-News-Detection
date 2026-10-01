import type { ReactNode } from "react";

interface Props {
  title: string;
  icon?: ReactNode;
  subtitle?: string;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
  style?: React.CSSProperties;
}

export function SectionCard({ title, icon, subtitle, actions, children, className = "", style }: Props) {
  return (
    <section className={`card ${className}`} style={style}>
      <header className="card__head">
        <div className="card__title">
          {icon && <span className="card__icon">{icon}</span>}
          <div>
            <h3>{title}</h3>
            {subtitle && <p className="card__subtitle">{subtitle}</p>}
          </div>
        </div>
        {actions && <div className="card__actions">{actions}</div>}
      </header>
      <div className="card__body">{children}</div>
    </section>
  );
}
