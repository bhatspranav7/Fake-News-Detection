import type {
  AnalysisResult,
  AnalyzeRequest,
  ExamplesResponse,
  FeedbackRequest,
  Health,
  HistoryResponse,
  ModelMetricsResponse,
  StepEvent,
  UsageMetrics,
} from "./types";

export const API_URL: string = (
  (import.meta.env.VITE_API_URL as string | undefined) ?? "http://127.0.0.1:8000"
).replace(/\/+$/, "");

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function parseError(res: Response): Promise<string> {
  try {
    const body = (await res.json()) as { detail?: unknown; message?: unknown };
    if (typeof body.detail === "string") return body.detail;
    if (typeof body.message === "string") return body.message;
    if (body.detail) return JSON.stringify(body.detail);
  } catch {
    /* ignore */
  }
  return `${res.status} ${res.statusText || "Request failed"}`;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    });
  } catch (err) {
    throw new ApiError(
      `Cannot reach the VeriFact API at ${API_URL}. Is the backend running?`,
      0,
    );
  }
  if (!res.ok) throw new ApiError(await parseError(res), res.status);
  return (await res.json()) as T;
}

export const api = {
  health: (signal?: AbortSignal) => request<Health>("/health", { signal }),

  analyze: (body: AnalyzeRequest, signal?: AbortSignal) =>
    request<AnalysisResult>("/analyze", {
      method: "POST",
      body: JSON.stringify(body),
      signal,
    }),

  feedback: (body: FeedbackRequest) =>
    request<{ ok: boolean }>("/feedback", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  history: (limit = 20) => request<HistoryResponse>(`/history?limit=${limit}`),

  analysis: (id: string) =>
    request<AnalysisResult>(`/analysis/${encodeURIComponent(id)}`),

  examples: () => request<ExamplesResponse>("/examples"),

  modelMetrics: () => request<ModelMetricsResponse>("/metrics/models"),

  usageMetrics: () => request<UsageMetrics>("/metrics/usage"),
};

export interface StreamHandlers {
  onStep: (step: StepEvent) => void;
  onResult: (result: AnalysisResult) => void;
  onError: (message: string) => void;
}

/**
 * POST /analyze/stream and consume the Server-Sent-Events body manually
 * (EventSource cannot POST). Resolves when the stream closes.
 */
export async function analyzeStream(
  body: AnalyzeRequest,
  handlers: StreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}/analyze/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify(body),
      signal,
    });
  } catch (err) {
    if ((err as Error).name === "AbortError") return;
    handlers.onError(
      `Cannot reach the VeriFact API at ${API_URL}. Is the backend running?`,
    );
    return;
  }

  if (!res.ok) {
    handlers.onError(await parseError(res));
    return;
  }
  if (!res.body) {
    handlers.onError("Streaming response had no body.");
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let gotResult = false;

  const dispatch = (eventName: string, dataLines: string[]) => {
    if (dataLines.length === 0) return;
    const raw = dataLines.join("\n");
    let data: unknown;
    try {
      data = JSON.parse(raw);
    } catch {
      return;
    }
    switch (eventName) {
      case "step":
        handlers.onStep(data as StepEvent);
        break;
      case "result":
        gotResult = true;
        handlers.onResult(data as AnalysisResult);
        break;
      case "error": {
        const d = data as { detail?: string };
        handlers.onError(d.detail ?? "Analysis failed.");
        break;
      }
      default:
        break;
    }
  };

  const processBlock = (block: string) => {
    let eventName = "message";
    const dataLines: string[] = [];
    for (const line of block.split(/\r?\n/)) {
      if (!line || line.startsWith(":")) continue;
      const idx = line.indexOf(":");
      const field = idx === -1 ? line : line.slice(0, idx);
      let value = idx === -1 ? "" : line.slice(idx + 1);
      if (value.startsWith(" ")) value = value.slice(1);
      if (field === "event") eventName = value;
      else if (field === "data") dataLines.push(value);
    }
    dispatch(eventName, dataLines);
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let sep: number;
      // Event blocks are separated by a blank line.
      while ((sep = buffer.search(/\r?\n\r?\n/)) !== -1) {
        const match = buffer.slice(sep).match(/^\r?\n\r?\n/);
        const block = buffer.slice(0, sep);
        buffer = buffer.slice(sep + (match ? match[0].length : 2));
        processBlock(block);
      }
    }
    buffer += decoder.decode();
    if (buffer.trim()) processBlock(buffer);
  } catch (err) {
    if ((err as Error).name === "AbortError") return;
    handlers.onError((err as Error).message || "Stream interrupted.");
    return;
  }

  if (!gotResult && !signal?.aborted) {
    handlers.onError("Stream ended before a result was produced.");
  }
}
