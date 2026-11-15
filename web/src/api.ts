// The service's one endpoint, POST /api/answer, read as a stream (D-050, D-051).

import { SseParser } from "./sse";

export interface Source {
  n: number;
  title: string;
  url: string;
}

export interface Done {
  cited: number[];
  declined: boolean;
  complete: boolean;
  cached: boolean;
  cost_usd: number;
  tokens: { input: number; output: number };
  timings_ms: {
    embed: number | null;
    search: number | null;
    retrieval: number;
    first_text: number | null;
    total: number;
  };
}

export type AnswerEvent =
  | { type: "sources"; articles: Source[] }
  | { type: "delta"; text: string }
  | { type: "done"; done: Done }
  | { type: "error"; message: string };

export function toAnswerEvent(event: string, data: string): AnswerEvent | null {
  const payload = JSON.parse(data);
  switch (event) {
    case "sources":
      return { type: "sources", articles: payload.articles };
    case "delta":
      return { type: "delta", text: payload.text };
    case "done":
      return { type: "done", done: payload };
    case "error":
      return { type: "error", message: payload.message };
    default:
      return null; // an event this page doesn't know yet
  }
}

/** Ask a question; `onEvent` hears each event as it arrives. A POST, not EventSource, which
 * only sends GET and would put the question in the URL. */
export async function ask(
  question: string,
  onEvent: (event: AnswerEvent) => void,
  signal: AbortSignal,
): Promise<void> {
  const response = await fetch("/api/answer", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
    signal,
  });
  if (!response.ok || response.body === null) {
    const message =
      response.status === 422
        ? "Please type a question, up to 8,000 characters."
        : `The service answered with status ${response.status}.`;
    onEvent({ type: "error", message });
    return;
  }
  // TextDecoderStream also keeps a character whose bytes are split across two chunks whole.
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  const parser = new SseParser();
  for (;;) {
    const { value, done } = await reader.read();
    if (done) return;
    for (const { event, data } of parser.push(value)) {
      const parsed = toAnswerEvent(event, data);
      if (parsed !== null) onEvent(parsed);
    }
  }
}
