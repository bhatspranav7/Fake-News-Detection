import { AlertOctagon, Info, X } from "lucide-react";

interface Props {
  message: string;
  kind?: "error" | "info" | "warning";
  onClose?: () => void;
}

export function ErrorBanner({ message, kind = "error", onClose }: Props) {
  return (
    <div className={`banner banner--${kind}`} role={kind === "error" ? "alert" : "status"}>
      {kind === "error" ? <AlertOctagon size={18} /> : <Info size={18} />}
      <span className="banner__text">{message}</span>
      {onClose && (
        <button className="icon-btn" onClick={onClose} aria-label="Dismiss">
          <X size={16} />
        </button>
      )}
    </div>
  );
}
