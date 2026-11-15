// The answer cites articles inline as [1], [2], ... (D-023), and the model marks UI names in
// **bold**. The page turns citations into links and bold into bold; nothing else is Markdown.

export type Part = string | number | { bold: string };

/** The answer's text, split into plain text, bold text and citation numbers:
 *  "Click **Save** [2]."  ->  ["Click ", { bold: "Save" }, " ", 2, "."] */
export function splitAnswer(text: string): Part[] {
  const parts: Part[] = [];
  let last = 0;
  for (const match of text.matchAll(/\*\*(.+?)\*\*|\[(\d+)\]/g)) {
    const start = match.index ?? 0;
    if (start > last) parts.push(text.slice(last, start));
    parts.push(match[2] !== undefined ? Number(match[2]) : { bold: match[1] });
    last = start + match[0].length;
  }
  if (last < text.length) parts.push(text.slice(last));
  return parts;
}

export const DECLINE = "NOT_FOUND";

/** While streaming, text that could still turn out to be the decline isn't shown. */
export function mayBeDecline(text: string): boolean {
  const trimmed = text.trim();
  return trimmed.length > 0 && DECLINE.startsWith(trimmed);
}
