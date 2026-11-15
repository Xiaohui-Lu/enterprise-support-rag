// Server-Sent Events, parsed from text that arrives in arbitrary pieces (D-051).
//
// A network chunk doesn't line up with an event: one chunk can end halfway through an event,
// or hold several. So the parser keeps what it has received, and an event is complete only
// once its blank line has arrived:
//
//   push("event: delta\ndata: {\"te")      ->  []
//   push("xt\": \"Hi\"}\n\nevent: do")      ->  [{ event: "delta", data: "{\"text\": \"Hi\"}" }]

export interface SseEvent {
  event: string;
  data: string;
}

export class SseParser {
  private buffer = "";

  push(text: string): SseEvent[] {
    this.buffer += text.replace(/\r\n?/g, "\n");
    const events: SseEvent[] = [];
    let end = this.buffer.indexOf("\n\n");
    while (end !== -1) {
      const block = this.buffer.slice(0, end);
      this.buffer = this.buffer.slice(end + 2);
      let event = "message";
      const data: string[] = [];
      for (const line of block.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data.push(line.slice(5).replace(/^ /, ""));
      }
      if (data.length > 0) events.push({ event, data: data.join("\n") });
      end = this.buffer.indexOf("\n\n");
    }
    return events;
  }
}
