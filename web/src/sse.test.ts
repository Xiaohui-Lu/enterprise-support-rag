import { describe, expect, it } from "vitest";

import { toAnswerEvent } from "./api";
import { mayBeDecline, splitAnswer } from "./citations";
import { SseParser } from "./sse";

describe("SseParser", () => {
  it("waits for the blank line before an event is complete", () => {
    const parser = new SseParser();
    expect(parser.push('event: delta\ndata: {"te')).toEqual([]);
    expect(parser.push('xt": "Hi"}\n\nevent: do')).toEqual([
      { event: "delta", data: '{"text": "Hi"}' },
    ]);
  });

  it("returns every event a chunk completes, in order", () => {
    const parser = new SseParser();
    const events = parser.push("event: a\ndata: 1\n\nevent: b\ndata: 2\n\n");
    expect(events.map((e) => e.event)).toEqual(["a", "b"]);
  });

  it("treats CRLF line endings like LF", () => {
    expect(new SseParser().push("event: a\r\ndata: 1\r\n\r\n")).toEqual([
      { event: "a", data: "1" },
    ]);
  });

  it("splits a stream cut at every possible point the same way", () => {
    const stream = 'event: sources\ndata: {"articles": []}\n\nevent: delta\ndata: {"text": "ok"}\n\n';
    for (let cut = 1; cut < stream.length; cut++) {
      const parser = new SseParser();
      const events = [...parser.push(stream.slice(0, cut)), ...parser.push(stream.slice(cut))];
      expect(events.map((e) => e.event)).toEqual(["sources", "delta"]);
    }
  });
});

describe("toAnswerEvent", () => {
  it("reads the service's events and skips unknown ones", () => {
    expect(toAnswerEvent("delta", '{"text": "Go"}')).toEqual({ type: "delta", text: "Go" });
    expect(toAnswerEvent("error", '{"message": "x"}')).toEqual({ type: "error", message: "x" });
    expect(toAnswerEvent("ping", "{}")).toBeNull();
  });
});

describe("citations", () => {
  it("splits the text around each citation number", () => {
    expect(splitAnswer("Go to Settings [2], then save [1].")).toEqual([
      "Go to Settings ",
      2,
      ", then save ",
      1,
      ".",
    ]);
  });

  it("leaves text without citations alone", () => {
    expect(splitAnswer("No citations here.")).toEqual(["No citations here."]);
  });

  it("marks bold text, and leaves a bold that hasn't closed yet as text", () => {
    expect(splitAnswer("Click **Save** [2].")).toEqual(["Click ", { bold: "Save" }, " ", 2, "."]);
    expect(splitAnswer("Click **Sa")).toEqual(["Click **Sa"]);
  });

  it("holds back text that could still be the decline", () => {
    expect(mayBeDecline("NOT_")).toBe(true);
    expect(mayBeDecline("NOT_FOUND")).toBe(true);
    expect(mayBeDecline("Not available")).toBe(false);
    expect(mayBeDecline("")).toBe(false);
  });
});
