import type { Source } from "./api";

/** The articles the model read, numbered as it saw them; the cited ones stand out. */
export function Sources({ sources, cited }: { sources: Source[]; cited: number[] }) {
  return (
    <details className="sources" open>
      <summary>
        Articles it read ({sources.length}){cited.length > 0 && `, ${new Set(cited).size} cited`}
      </summary>
      <ol>
        {sources.map((source) => (
          <li key={source.n} className={cited.includes(source.n) ? "cited" : undefined}>
            <span className="number">{source.n}</span>
            <a href={source.url} target="_blank" rel="noreferrer">
              {source.title}
            </a>
          </li>
        ))}
      </ol>
    </details>
  );
}
