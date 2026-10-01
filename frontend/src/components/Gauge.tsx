import { useEffect, useState } from "react";
import { clamp01, probColor } from "../utils";

interface Props {
  /** 0..1 */
  value: number;
  label?: string;
  size?: number;
  color?: string;
}

/** Semicircular gauge showing a 0..1 value, animated on mount. */
export function Gauge({ value, label = "fake probability", size = 220, color }: Props) {
  const target = clamp01(value);
  const [v, setV] = useState(0);

  useEffect(() => {
    const id = window.requestAnimationFrame(() => setV(target));
    return () => window.cancelAnimationFrame(id);
  }, [target]);

  const stroke = 16;
  const r = (size - stroke) / 2;
  const cx = size / 2;
  const cy = size / 2;
  const circumference = Math.PI * r; // half circle
  const offset = circumference * (1 - v);
  const c = color ?? probColor(target);

  // needle position
  const angle = Math.PI * (1 - v);
  const nx = cx + r * Math.cos(angle);
  const ny = cy - r * Math.sin(angle);

  return (
    <div className="gauge" style={{ width: size, height: size / 2 + 36 }}>
      <svg viewBox={`0 0 ${size} ${size / 2 + 12}`} width={size} height={size / 2 + 12} aria-hidden="true">
        <defs>
          <linearGradient id="gauge-track" x1="0" x2="1">
            <stop offset="0%" stopColor="var(--success)" stopOpacity="0.35" />
            <stop offset="50%" stopColor="var(--warning)" stopOpacity="0.35" />
            <stop offset="100%" stopColor="var(--danger)" stopOpacity="0.35" />
          </linearGradient>
        </defs>
        <path
          d={`M ${stroke / 2} ${cy} A ${r} ${r} 0 0 1 ${size - stroke / 2} ${cy}`}
          fill="none"
          stroke="url(#gauge-track)"
          strokeWidth={stroke}
          strokeLinecap="round"
        />
        <path
          d={`M ${stroke / 2} ${cy} A ${r} ${r} 0 0 1 ${size - stroke / 2} ${cy}`}
          fill="none"
          stroke={c}
          strokeWidth={stroke}
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          style={{ transition: "stroke-dashoffset 1.1s cubic-bezier(.22,1,.36,1), stroke .4s" }}
        />
        <circle
          cx={nx}
          cy={ny}
          r={stroke / 2 + 2}
          fill="var(--bg-elev)"
          stroke={c}
          strokeWidth={3}
          style={{ transition: "cx 1.1s cubic-bezier(.22,1,.36,1), cy 1.1s cubic-bezier(.22,1,.36,1)" }}
        />
      </svg>
      <div className="gauge__value" style={{ color: c }}>
        {Math.round(target * 100)}
        <span className="gauge__unit">%</span>
      </div>
      <div className="gauge__label">{label}</div>
    </div>
  );
}
