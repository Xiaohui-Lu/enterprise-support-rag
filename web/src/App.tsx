import { type FormEvent, type KeyboardEvent, useRef, useState } from "react";

import { Answer } from "./Answer";
import { ask, type Done, type Source } from "./api";
import { Sources } from "./Sources";

type Status = "idle" | "searching" | "writing" | "finished";

const EXAMPLES = [
  "Can I change the font size of my blog posts on mobile?",
  "How do I connect a domain I bought somewhere else?",
  "Why can't I find my site on Google?",
];

export function App() {
  const [question, setQuestion] = useState("");
  const [asked, setAsked] = useState("");
  const [status, setStatus] = useState<Status>("idle");
  const [sources, setSources] = useState<Source[]>([]);
  const [text, setText] = useState("");
  const [done, setDone] = useState<Done | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [stopped, setStopped] = useState(false);
  const controller = useRef<AbortController | null>(null);
  const busy = status === "searching" || status === "writing";

  async function submit(text: string) {
    const trimmed = text.trim();
    if (trimmed === "" || busy) return;
    const abort = new AbortController();
    controller.current = abort;
    setAsked(trimmed);
    setStatus("searching");
    setSources([]);
    setText("");
    setDone(null);
    setError(null);
    setStopped(false);
    try {
      await ask(
        trimmed,
        (event) => {
          if (event.type === "sources") {
            setSources(event.articles);
            setStatus("writing");
          } else if (event.type === "delta") {
            setText((so_far) => so_far + event.text);
          } else if (event.type === "done") {
            setDone(event.done);
          } else {
            setError(event.message);
          }
        },
        abort.signal,
      );
    } catch {
      if (abort.signal.aborted) setStopped(true);
      else setError("Couldn't reach the service. Is `srag serve` running?");
    } finally {
      setStatus("finished");
    }
  }

  function onSubmit(event: FormEvent) {
    event.preventDefault();
    void submit(question);
  }

  function onKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void submit(question);
    }
  }

  return (
    <main className="page">
      <header className="header">
        <h1>Support assistant</h1>
        <p className="lede">
          Answers from a 6,221-article help center (WixQA), citing the articles it used. Single-pass
          RAG: dense search finds the top 10 articles, and Claude Haiku 5.5 answers from them.
        </p>
      </header>

      <form className="ask" onSubmit={onSubmit}>
        <label htmlFor="question" className="visually-hidden">
          Your question
        </label>
        <textarea
          id="question"
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          onKeyDown={onKeyDown}
          placeholder="Ask about your site, domain, store or blog…"
          rows={3}
          maxLength={8000}
        />
        <div className="ask-row">
          <div className="examples">
            {EXAMPLES.map((example) => (
              <button
                key={example}
                type="button"
                className="chip"
                disabled={busy}
                onClick={() => {
                  setQuestion(example);
                  void submit(example);
                }}
              >
                {example}
              </button>
            ))}
          </div>
          {busy ? (
            <button type="button" className="button secondary" onClick={() => controller.current?.abort()}>
              Stop
            </button>
          ) : (
            <button type="submit" className="button" disabled={question.trim() === ""}>
              Ask
            </button>
          )}
        </div>
      </form>

      {status !== "idle" && (
        <section className="result" aria-live="polite">
          <p className="asked">{asked}</p>
          {status === "searching" && <p className="muted">Searching the help center…</p>}
          {error !== null && <p className="error">{error}</p>}
          {stopped && <p className="muted">Stopped.</p>}
          {status !== "searching" && error === null && (
            <Answer text={text} sources={sources} done={done} streaming={status === "writing"} />
          )}
          {done !== null && <Meta done={done} />}
          {sources.length > 0 && <Sources sources={sources} cited={done?.cited ?? []} />}
        </section>
      )}
    </main>
  );
}

function Meta({ done }: { done: Done }) {
  const seconds = (ms: number | null) => (ms === null ? "n/a" : `${(ms / 1000).toFixed(1)} s`);
  return (
    <p className="meta">
      {seconds(done.timings_ms.total)} in all · first words at {seconds(done.timings_ms.first_text)}{" "}
      · ${done.cost_usd.toFixed(4)}
      {done.cached && <span className="tag">from cache, free</span>}
      {!done.complete && <span className="tag warn">cut off</span>}
    </p>
  );
}
