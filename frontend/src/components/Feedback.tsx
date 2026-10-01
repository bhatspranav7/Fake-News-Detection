import { useState } from "react";
import { Loader2, ThumbsDown, ThumbsUp } from "lucide-react";
import { api } from "../api";

interface Props {
  analysisId: string;
}

type State = "idle" | "sending" | "done" | "error";

export function Feedback({ analysisId }: Props) {
  const [state, setState] = useState<State>("idle");
  const [choice, setChoice] = useState<"correct" | "incorrect" | null>(null);
  const [comment, setComment] = useState("");
  const [showComment, setShowComment] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const send = async (rating: "correct" | "incorrect") => {
    setChoice(rating);
    setState("sending");
    setError(null);
    try {
      await api.feedback({ analysis_id: analysisId, rating, comment: comment.trim() || undefined });
      setState("done");
    } catch (e) {
      setState("error");
      setError((e as Error).message);
    }
  };

  if (state === "done") {
    return (
      <div className="feedback feedback--done">
        <span>Thanks — your feedback helps calibrate VeriFact.</span>
        <span className={`badge badge--${choice === "correct" ? "supported" : "refuted"}`}>{choice}</span>
      </div>
    );
  }

  return (
    <div className="feedback">
      <div className="feedback__row">
        <span className="feedback__q">Was this verdict right?</span>
        <div className="feedback__btns">
          <button className="btn btn--ghost" onClick={() => void send("correct")} disabled={state === "sending"} aria-label="Verdict was correct">
            {state === "sending" && choice === "correct" ? <Loader2 size={16} className="spin" /> : <ThumbsUp size={16} />}
            Yes
          </button>
          <button className="btn btn--ghost" onClick={() => void send("incorrect")} disabled={state === "sending"} aria-label="Verdict was incorrect">
            {state === "sending" && choice === "incorrect" ? <Loader2 size={16} className="spin" /> : <ThumbsDown size={16} />}
            No
          </button>
          <button className="btn btn--link" onClick={() => setShowComment((v) => !v)}>
            {showComment ? "Hide comment" : "Add a comment"}
          </button>
        </div>
      </div>
      {showComment && (
        <textarea
          className="input feedback__comment"
          rows={2}
          placeholder="Optional: tell us what we got wrong…"
          value={comment}
          onChange={(e) => setComment(e.target.value)}
          maxLength={500}
        />
      )}
      {error && <p className="feedback__err">Could not send feedback: {error}</p>}
    </div>
  );
}
