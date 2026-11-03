"""Hand labels for validating the judge (D-005, D-027).

The sample is drawn once from whole dev questions and committed with the answer texts, so
the labels are self-contained ground truth. The questions left out are a reserve: if the
rubric is revised after reading disagreements, the revised judge is checked on fresh labels
from the reserve, never on the labels it was fixed against.

Labeling is blind: an item shows the question, the expert reference and the answer, never
the strategy, the run or any judge grade. Each label is appended as soon as it's given, so
a session can stop and resume.
"""

from __future__ import annotations

import json
import random
from collections import Counter, deque
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from support_rag import config
from support_rag.judge import CORRECTNESS_RUBRIC, GROUNDED_RUBRIC
from support_rag.records import Article, Question
from support_rag.runs import load_config, load_results

SAMPLE_PATH = config.LABELS_DIR / "sample.json"
LABELS_PATH = config.LABELS_DIR / "labels.jsonl"

LABEL_SEED = 20261102
TARGET_ITEMS = 80
GROUNDED_ITEMS = 20
# The pilot answer whose grade was discussed in the rubric lesson (D-025): already seen.
EXCLUDED = ("ew-060",)

CORRECTNESS_KEYS = {"c": "correct", "p": "partial", "i": "incorrect"}
GROUNDED_KEYS = {"y": "yes", "p": "partly", "n": "no"}

HELP = f"""\
Grade each answer against the expert reference, by the judge's own rubric:

{CORRECTNESS_RUBRIC}

Some items also ask for groundedness, and show the articles the answer cites:

{GROUNDED_RUBRIC}

Type c, p or i, optionally followed by a note: "p skips the publish step".
u goes back to the previous item; q quits. Every label is saved as you go."""


def draw_sample(
    answers_by_strategy: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    target: int = TARGET_ITEMS,
    n_grounded: int = GROUNDED_ITEMS,
    exclude: Sequence[str] = EXCLUDED,
    seed: int = LABEL_SEED,
) -> dict[str, Any]:
    """Take whole questions in random order until the sample holds `target` answers.

    Declines are left out: they never reach the judge (D-025), so there's nothing to agree on.
    """
    by_question: dict[str, list[tuple[str, Mapping[str, Any]]]] = {}
    for strategy in sorted(answers_by_strategy):
        for answer in answers_by_strategy[strategy]:
            by_question.setdefault(answer["question_id"], []).append((strategy, answer))
    if any(len(pairs) != len(answers_by_strategy) for pairs in by_question.values()):
        raise ValueError("the answer runs don't cover the same questions")

    rng = random.Random(seed)
    question_ids = sorted(set(by_question) - set(exclude))
    rng.shuffle(question_ids)
    labeled: list[str] = []
    chosen: list[tuple[str, Mapping[str, Any]]] = []
    for question_id in question_ids:
        if len(chosen) >= target:
            break
        labeled.append(question_id)
        chosen += [(s, a) for s, a in by_question[question_id] if not a["declined"]]
    reserve = question_ids[len(labeled) :]

    rng.shuffle(chosen)
    with_citations = [n for n, (_, answer) in enumerate(chosen) if answer["cited_ids"]]
    grounded = set(rng.sample(with_citations, min(n_grounded, len(with_citations))))
    items = [
        {
            "item": n + 1,
            "question_id": answer["question_id"],
            "strategy": strategy,
            "text": answer["text"],
            "context_ids": list(answer["context_ids"]),
            "cited_ids": list(answer["cited_ids"]),
            "check_grounded": n in grounded,
        }
        for n, (strategy, answer) in enumerate(chosen)
    ]
    return {
        "seed": seed,
        "excluded": list(exclude),
        "counts": {
            "items": len(items),
            **{f"items_{s}": n for s, n in sorted(Counter(i["strategy"] for i in items).items())},
            "grounded_checks": len(grounded),
            "labeled_questions": len(labeled),
            "reserve_questions": len(reserve),
        },
        "labeled_questions": sorted(labeled),
        "reserve_questions": sorted(reserve),
        "items": items,
    }


def write_sample(answer_runs: Sequence[Path], path: Path = SAMPLE_PATH) -> dict[str, Any]:
    """Draw the sample from one full dev_real answer run per strategy. Runs once."""
    if path.exists():
        raise FileExistsError(
            f"{path} already exists. The sample is drawn once; labels refer to its items."
        )
    configs = [load_config(run) for run in answer_runs]
    strategies = [c.get("strategy") for c in configs]
    if any(c.get("kind") != "answers" or c.get("split") != "dev_real" for c in configs) or len(
        set(strategies)
    ) != len(strategies):
        raise ValueError("give one dev_real answer run per strategy")
    answers = {
        c["strategy"]: load_results(run) for c, run in zip(configs, answer_runs, strict=True)
    }
    sample = {
        "version": 1,
        "answer_runs": {
            c["strategy"]: {k: c[k] for k in ("run_id", "model", "effort", "prompt_version")}
            for c in configs
        },
        **draw_sample(answers),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(sample, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    return sample


def load_sample(path: Path = SAMPLE_PATH) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `srag label-sample` first.")
    return json.loads(path.read_text(encoding="utf-8"))


def load_labels(path: Path = LABELS_PATH) -> dict[int, dict[str, Any]]:
    """Labels by item. A later label for the same item replaces an earlier one (undo)."""
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        labels = [json.loads(line) for line in f if line.strip()]
    return {label["item"]: label for label in labels}


def append_label(path: Path, label: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(label, ensure_ascii=False) + "\n")


def adjudication_items(sample: Mapping[str, Any], panel: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The sample items the panel didn't settle (D-028), in sample order. Groundedness is
    asked only where the panel also split on it and the answer cites articles."""
    by_item = {p["item"]: p for p in panel["items"]}
    return [
        {
            **item,
            "check_grounded": bool(item["cited_ids"]) and by_item[item["item"]]["grounded"] is None,
        }
        for item in sample["items"]
        if by_item[item["item"]]["correctness"] is None
    ]


def label_session(
    items: Sequence[Mapping[str, Any]],
    questions: Mapping[str, Question],
    articles_by_id: Mapping[str, Article],
    *,
    labels_path: Path = LABELS_PATH,
    ask: Callable[[str], str] = input,
    show: Callable[[str], None] = print,
) -> int:
    """Label the given items not labeled yet, in order. Returns how many were labeled."""
    done = load_labels(labels_path)
    queue = deque(item for item in items if item["item"] not in done)
    history: list[Mapping[str, Any]] = []
    show(HELP)
    labeled = 0
    while queue:
        item = queue[0]
        position = len(items) - len(queue) + 1
        show(_render(item, questions[item["question_id"]], position=position, total=len(items)))
        key, note = _ask(ask, "correctness [c/p/i + note, u, q]> ", {*CORRECTNESS_KEYS, "u", "q"})
        if key == "q":
            break
        if key == "u":
            if history:
                queue.appendleft(history.pop())
            else:
                show("Nothing to undo in this session.")
            continue
        grounded = None
        if item["check_grounded"]:
            show(_render_cited(item, articles_by_id))
            g_key, g_note = _ask(ask, "grounded [y/p/n + note, q]> ", {*GROUNDED_KEYS, "q"})
            if g_key == "q":
                break
            grounded = GROUNDED_KEYS[g_key]
            note = "; ".join(n for n in (note, g_note) if n)
        append_label(
            labels_path,
            {
                "item": item["item"],
                "correctness": CORRECTNESS_KEYS[key],
                "grounded": grounded,
                "note": note,
                "labeled_at": datetime.now().isoformat(timespec="seconds"),
            },
        )
        history.append(queue.popleft())
        labeled += 1
    labels = load_labels(labels_path)
    show(f"\n{sum(item['item'] in labels for item in items)} of {len(items)} items labeled.")
    return labeled


def _ask(ask: Callable[[str], str], prompt: str, keys: set[str]) -> tuple[str, str]:
    """Ask until the reply starts with a valid key. Ctrl+C or end of input means quit."""
    while True:
        try:
            reply = ask(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            return "q", ""
        key, _, note = reply.partition(" ")
        if key.lower() in keys:
            return key.lower(), note.strip()


def _render(item: Mapping[str, Any], question: Question, *, position: int, total: int) -> str:
    rule = "-" * 72
    return (
        f"\n{rule}\nItem {position} of {total}\n{rule}\n\n"
        f"QUESTION\n{question.question}\n\n"
        f"REFERENCE ANSWER (support expert)\n{question.answer}\n\n"
        f"ANSWER TO GRADE\n{item['text']}\n"
    )


def _render_cited(item: Mapping[str, Any], articles_by_id: Mapping[str, Article]) -> str:
    # The answer cites articles by their number in its own prompt: [1] is context_ids[0].
    numbers = {article_id: n for n, article_id in enumerate(item["context_ids"], start=1)}
    cited = [
        f"[{numbers[a]}] {articles_by_id[a].title}\n{articles_by_id[a].contents}"
        for a in item["cited_ids"]
    ]
    return "\nCITED ARTICLES\n\n" + "\n\n".join(cited) + "\n"
