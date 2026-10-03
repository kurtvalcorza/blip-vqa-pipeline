"""Corpus-level VQA metrics and two non-neural baselines.

The per-question helpers (`normalize_answer`, `vqa_accuracy`, `exact_match`, `anls`) live in `pipeline.py`.
This
module averages them over a dataset and adds two baselines a fine-tuned model must beat: **constant answer**
(the same string for every question — `unanswerable` scores exactly the share of questions whose annotators
gave it, the VizWiz-specific trap) and **question-prefix majority** (the most frequent training answer among
questions that start with the same two words, e.g. `what color` → `white`; the global training majority when
the prefix is unseen — a lookup with no model that knows nothing about the image).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from .pipeline import anls, exact_match, normalize_answer, vqa_accuracy

PREFIX_WORDS = 2
ABSTAIN_ANSWER = "unanswerable"
METRIC_DEFINITIONS = {
    "vqa_accuracy": (
        "mean over questions of min(number of accepted answers the normalised prediction equals / 3, 1) — "
        "the "
        "VQA v2 / VizWiz convention with ten human answers per question; in 0..1"
    ),
    "exact_match": (
        "fraction of questions whose normalised prediction equals any normalised accepted answer; in 0..1"
    ),
    "anls": (
        "mean over questions of the Average Normalised Levenshtein Similarity between the prediction and its "
        "best-matching accepted answer (a similarity below 0.5 scores 0); in 0..1"
    ),
    "majority_match": (
        "fraction of questions whose normalised prediction equals the question's majority answer (the most "
        "frequent of its accepted answers) — stricter than exact_match on VizWiz, where `unanswerable` sits "
        "among the ten answers of most questions; in 0..1"
    ),
}


def majority_answer(answers: Sequence[str]) -> str:
    """The most frequent normalised answer (first on ties)."""
    counts = Counter(normalize_answer(a) for a in answers)
    order = list(counts)
    return max(counts, key=lambda a: (counts[a], -order.index(a)))


def vqa_metrics(predictions: Sequence[str], golds: Sequence[Sequence[str]]) -> dict[str, Any]:
    """Mean VQA accuracy, exact-match rate and mean ANLS over parallel predictions and accepted-answer
    lists."""
    if len(predictions) != len(golds):
        raise ValueError(f"{len(predictions)} predictions but {len(golds)} gold lists")
    if not predictions:
        raise ValueError("no predictions to score")
    pairs = list(zip(predictions, golds, strict=True))
    return {
        "n": len(predictions),
        "vqa_accuracy": sum(vqa_accuracy(p, g) for p, g in pairs) / len(pairs),
        "exact_match": sum(exact_match(p, g) for p, g in pairs) / len(pairs),
        "anls": sum(anls(p, g) for p, g in pairs) / len(pairs),
        "majority_match": sum(normalize_answer(p) == majority_answer(g) for p, g in pairs) / len(pairs),
        "empty_rate": sum(1 for p in predictions if not p.strip()) / len(predictions),
        "definitions": dict(METRIC_DEFINITIONS),
    }


def _golds(records: Sequence[Mapping[str, Any]]) -> list[list[str]]:
    return [[str(a) for a in r["answers"]] for r in records]


def constant_answer_baseline(
    records: Sequence[Mapping[str, Any]], answer: str = "unanswerable"
) -> dict[str, Any]:
    """The same answer for every question."""
    result = vqa_metrics([answer] * len(records), _golds(records))
    result["baseline"] = f"constant answer {answer!r}"
    return result


def question_prefix(question: str, *, words: int = PREFIX_WORDS) -> str:
    return " ".join(normalize_answer(question).split()[:words])


def question_prefix_answers(train: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Per question prefix, the most frequent majority answer over the training records; `""` holds the
    global majority used for unseen prefixes."""
    per_prefix: dict[str, Counter[str]] = {}
    overall: Counter[str] = Counter()
    for record in train:
        answer = majority_answer(record["answers"])
        per_prefix.setdefault(question_prefix(record["question"]), Counter())[answer] += 1
        overall[answer] += 1
    table = {prefix: counts.most_common(1)[0][0] for prefix, counts in per_prefix.items()}
    table[""] = overall.most_common(1)[0][0] if overall else ""
    return table


def question_prefix_baseline(
    train: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Answer every question with the training majority answer for its two-word prefix (no model, no
    image)."""
    if not train:
        raise ValueError("the question-prefix baseline needs training records")
    table = question_prefix_answers(train)
    predictions = [table.get(question_prefix(r["question"]), table[""]) for r in records]
    result = vqa_metrics(predictions, _golds(records))
    result["baseline"] = f"question-prefix majority ({len(table) - 1} prefixes seen in training)"
    return result


def _group_stats(rows: Sequence[tuple[str, list[str]]], abstain: str) -> dict[str, Any]:
    n = len(rows)
    abstained = [(p, g) for p, g in rows if normalize_answer(p) == abstain]
    content = [(p, g) for p, g in rows if normalize_answer(p) != abstain]
    return {
        "n": n,
        "abstain_share": len(abstained) / n,
        "vqa_accuracy": sum(vqa_accuracy(p, g) for p, g in rows) / n,
        "vqa_from_abstain": sum(vqa_accuracy(p, g) for p, g in abstained) / n,
        "content_n": len(content),
        "content_vqa_accuracy": (
            sum(vqa_accuracy(p, g) for p, g in content) / len(content) if content else None
        ),
    }


def answerability_breakdown(
    predictions: Sequence[str],
    records: Sequence[Mapping[str, Any]],
    *,
    abstain: str = ABSTAIN_ANSWER,
) -> dict[str, Any]:
    """How much of a VQA score comes from answering `abstain` (VizWiz's `unanswerable`).

    For the whole set, per `category` and per majority-answer group (questions whose majority answer is
    `abstain` vs the rest), it reports the share of `abstain` predictions, mean VQA accuracy, the part of that
    mean earned by the `abstain` predictions (`vqa_from_abstain`, so `vqa_accuracy - vqa_from_abstain` is the
    part earned by content answers), and the mean VQA accuracy over the content answers alone
    (`content_vqa_accuracy`, None when every prediction is `abstain`). A system that learns to answer
    `abstain` often can raise VQA accuracy on a category without answering any of its questions better; this
    breakdown separates the two (review VQA-M2)."""
    if len(predictions) != len(records):
        raise ValueError(f"{len(predictions)} predictions but {len(records)} records")
    if not predictions:
        raise ValueError("no predictions to score")
    word = normalize_answer(abstain)
    rows = [(str(p), _golds([r])[0], r) for p, r in zip(predictions, records, strict=True)]
    categories: dict[str, list[tuple[str, list[str]]]] = {}
    majority: dict[str, list[tuple[str, list[str]]]] = {}
    for prediction, golds, record in rows:
        categories.setdefault(str(record.get("category", "byod")), []).append((prediction, golds))
        group = f"majority {word}" if majority_answer(golds) == word else f"majority not {word}"
        majority.setdefault(group, []).append((prediction, golds))
    return {
        "abstain_answer": word,
        "overall": _group_stats([(p, g) for p, g, _ in rows], word),
        "by_category": {name: _group_stats(part, word) for name, part in sorted(categories.items())},
        "by_majority_answer": {name: _group_stats(part, word) for name, part in sorted(majority.items())},
    }
