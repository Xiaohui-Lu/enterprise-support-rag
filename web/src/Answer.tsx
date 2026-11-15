import type { Done, Source } from "./api";
import { mayBeDecline, splitAnswer } from "./citations";

interface Props {
  text: string;
  sources: Source[];
  done: Done | null;
  streaming: boolean;
}

/** The answer as it streams, with each [n] a link to the article it cites and **bold** bold. */
export function Answer({ text, sources, done, streaming }: Props) {
  if (done?.declined) {
    return (
      <p className="decline">
        I couldn't find this in the help center. The closest articles it searched are below.
      </p>
    );
  }
  if (streaming && (text === "" || mayBeDecline(text))) {
    return <p className="muted">Writing…</p>;
  }
  return (
    <div className="answer">
      {splitAnswer(text).map((part, i) =>
        typeof part === "number" ? (
          <Citation key={i} n={part} source={sources[part - 1]} />
        ) : typeof part === "string" ? (
          <span key={i}>{part}</span>
        ) : (
          <strong key={i}>{part.bold}</strong>
        ),
      )}
      {streaming && <span className="cursor" aria-hidden="true" />}
    </div>
  );
}

function Citation({ n, source }: { n: number; source: Source | undefined }) {
  if (source === undefined) return <span>[{n}]</span>; // a number the model made up
  return (
    <a className="citation" href={source.url} target="_blank" rel="noreferrer" title={source.title}>
      {n}
    </a>
  );
}
