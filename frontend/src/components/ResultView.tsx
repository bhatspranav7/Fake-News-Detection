import { Activity } from "lucide-react";
import type { AnalysisResult } from "../types";
import { AgentTimeline, type TimelineStep } from "./AgentTimeline";
import { ClaimsList } from "./ClaimsList";
import { Feedback } from "./Feedback";
import { LlmCard } from "./LlmCard";
import { ModelBreakdown } from "./ModelBreakdown";
import { SectionCard } from "./SectionCard";
import { SourceCard } from "./SourceCard";
import { TokenChips } from "./TokenChips";
import { VerdictCard } from "./VerdictCard";

interface Props {
  result: AnalysisResult;
  /** When true, the agent run summary card is shown (useful when the live timeline isn't visible). */
  showAgents?: boolean;
}

export function ResultView({ result, showAgents = true }: Props) {
  const hasLlm = result.llm !== null && (result.llm.summary || result.llm.red_flags.length > 0 || result.llm.reasoning);
  const hasClaims = result.claims && result.claims.length > 0;
  const agentSteps: TimelineStep[] = result.agents.map((a) => ({
    agent: a.name,
    status: a.status,
    summary: a.summary,
    elapsed_ms: a.elapsed_ms,
  }));

  return (
    <div className="result fade-in">
      <VerdictCard result={result} />

      <div className="grid grid--2">
        <ModelBreakdown ml={result.ml} />
        <TokenChips tokens={result.ml.top_tokens} styleFlags={result.ml.style_flags} />
      </div>

      {hasClaims && <ClaimsList claims={result.claims} />}

      <div className="grid grid--2">
        {hasLlm && result.llm && <LlmCard llm={result.llm} />}
        <SourceCard source={result.source} />
      </div>

      {showAgents && agentSteps.length > 0 && (
        <SectionCard title="Agent run" subtitle="What each agent did in this analysis" icon={<Activity size={18} />}>
          <AgentTimeline steps={agentSteps} compact />
        </SectionCard>
      )}

      <Feedback analysisId={result.id} />
    </div>
  );
}
