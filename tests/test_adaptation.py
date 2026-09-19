"""Offline tests for the VQA dataset contract, the pinned corpus readers, the corpus metrics and baselines,
BYOD loaders, JSONL export, artifact-manifest rejections and adapt() argument validation. Nothing here imports
torch, transformers or pyarrow; annotations and images come from injected fetchers and tiny PIL drawings."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from blip_vqa_pipeline import (
    ARTIFACT_FORMAT,
    CORPUS_FILE,
    DECODER_LAYERS,
    IMAGE_PINS,
    MODEL_ID,
    MODEL_REVISION,
    SAMPLE_SPLIT,
    WEIGHT_SHA256,
    BlipVQAPipeline,
    build_sample_dataset,
    check_split_disjoint,
    column_digest,
    constant_answer_baseline,
    dataset_digest,
    fetch_annotations,
    fetch_images,
    fetch_sample_dataset,
    load_byod_dataset,
    majority_answer,
    question_prefix_answers,
    question_prefix_baseline,
    split_dataset,
    validate_dataset,
    vqa_metrics,
    write_dataset_jsonl,
)
from blip_vqa_pipeline import pipeline as pl
from blip_vqa_pipeline import samples as sm

COLOURS = ["red", "green", "blue", "yellow", "white", "black"]


def _picture(path, colour, size=(96, 64)):
    image = Image.new("RGB", size, "gray")
    ImageDraw.Draw(image).rectangle([16, 12, 80, 52], fill=colour)
    image.save(path, format="JPEG")
    return path


def _annotations(n):
    rows = []
    for i in range(n):
        colour = COLOURS[i % len(COLOURS)]
        unanswerable = i % 3 == 0
        rows.append(
            {
                "question_id": f"VizWiz_val_{i:08d}",
                "question": "Can you tell me what this is?"
                if unanswerable
                else f"What color is the box number {i}?",
                "answers": ["unanswerable"] * 7 + ["blurry", "unanswerable", "unsuitable image"]
                if unanswerable
                else [colour] * 6 + ["unanswerable", colour, "dark " + colour, "unanswerable"],
                "category": "unanswerable" if unanswerable else "other",
            }
        )
    return rows


def _records(tmp_path, n=12, prefix="r"):
    out = []
    for i, row in enumerate(_annotations(n)):
        path = _picture(tmp_path / f"{prefix}{i}.jpg", COLOURS[i % len(COLOURS)])
        out.append(
            {
                "id": f"{prefix}{i:03d}",
                "image_id": row["question_id"],
                "image": str(path),
                "question": row["question"],
                "answers": row["answers"],
                "category": row["category"],
            }
        )
    return out


def _fake_pipeline():
    def runner(image, question, max_new_tokens):
        return {"answer": "unanswerable", "new_tokens": 1}

    return BlipVQAPipeline(runner, "cpu", "float32", "injected")


# --- corpus reader ----------------------------------------------------------------------------------


def test_pinned_corpus_constants():
    assert sm.CORPUS_REPO == "lmms-lab-encoder/VizWiz-VQA" and len(sm.CORPUS_REVISION) == 40
    assert CORPUS_FILE["path"].startswith("data/val-") and CORPUS_FILE["rows"] == 864
    assert len(CORPUS_FILE["sha256"]) == 64 and len(CORPUS_FILE["column_sha256"]) == 64
    assert len(IMAGE_PINS) == sm.SAMPLE_SIZE == 240 and sum(SAMPLE_SPLIT.values()) == 240
    assert all(len(digest) == 64 and size > 0 for digest, size in IMAGE_PINS.values())
    assert sm.IMAGE_HOST.startswith("https://vizwiz.cs.colorado.edu/")


def test_fetch_annotations_verifies_digest_and_caches(tmp_path, monkeypatch, forbid_model_imports):
    rows = _annotations(5)
    monkeypatch.setitem(CORPUS_FILE, "rows", 5)
    monkeypatch.setitem(CORPUS_FILE, "column_sha256", column_digest(rows))
    calls = []

    def fetcher():
        calls.append(1)
        return rows

    assert fetch_annotations(cache_dir=tmp_path, fetcher=fetcher) == rows
    assert fetch_annotations(cache_dir=tmp_path, fetcher=fetcher) == rows  # served from the cache
    assert len(calls) == 1
    with pytest.raises(ValueError, match="pinned"):
        fetch_annotations(
            cache_dir=tmp_path / "other", fetcher=lambda: rows[:4] + [{**rows[4], "question": "x"}]
        )
    monkeypatch.setitem(CORPUS_FILE, "rows", 99)
    with pytest.raises(ValueError, match="pinned 99"):
        fetch_annotations(cache_dir=tmp_path / "third", fetcher=fetcher)


def test_fetch_images_pins_every_file_and_spaces_requests(tmp_path, monkeypatch, forbid_model_imports):
    payloads = {
        f"VizWiz_val_{i:08d}": _picture(tmp_path / f"src{i}.jpg", COLOURS[i]).read_bytes() for i in range(3)
    }
    monkeypatch.setattr(
        sm,
        "IMAGE_PINS",
        {qid: (hashlib.sha256(data).hexdigest(), len(data)) for qid, data in payloads.items()},
    )
    calls, sleeps = [], []

    def fetcher(qid):
        calls.append(qid)
        return payloads[qid]

    paths = fetch_images(sorted(payloads), cache_dir=tmp_path / "cache", fetcher=fetcher, sleep=sleeps.append)
    assert sorted(paths) == sorted(payloads) and all(p.is_file() for p in paths.values())
    assert calls == sorted(payloads) and sleeps == [sm.FETCH_SPACING_SECONDS] * 2
    again = fetch_images(sorted(payloads), cache_dir=tmp_path / "cache", fetcher=fetcher, sleep=sleeps.append)
    assert again == paths and len(calls) == 3  # cached files are re-verified, not re-fetched
    with pytest.raises(ValueError, match="pinned"):
        fetch_images(["VizWiz_val_00000000"], cache_dir=tmp_path / "bad", fetcher=lambda qid: b"tampered")
    with pytest.raises(ValueError, match="not one of the"):
        fetch_images(["VizWiz_val_99999999"], cache_dir=tmp_path / "bad", fetcher=fetcher)


def test_fetch_with_backoff_honours_retry_after(monkeypatch, forbid_model_imports):
    import urllib.error

    attempts, sleeps = [], []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return b"ok"

    def fake_urlopen(url, timeout=0):
        attempts.append(url)
        if len(attempts) < 3:
            headers = {"Retry-After": "2"} if len(attempts) == 1 else {}
            raise urllib.error.HTTPError(url, 429 if len(attempts) == 1 else 503, "busy", headers, None)
        return Response()

    monkeypatch.setattr(sm.urllib.request, "urlopen", fake_urlopen)
    assert sm._fetch_with_backoff("https://example.invalid/x.jpg", sleep=sleeps.append) == b"ok"
    assert len(attempts) == 3 and sleeps == [2.0, 2.0]  # Retry-After honoured, then the doubled backoff

    def hard_fail(url, timeout=0):
        raise urllib.error.HTTPError(url, 404, "missing", {}, None)

    monkeypatch.setattr(sm.urllib.request, "urlopen", hard_fail)
    with pytest.raises(urllib.error.HTTPError):
        sm._fetch_with_backoff("https://example.invalid/y.jpg", sleep=sleeps.append)


def test_build_sample_dataset_is_seeded_image_disjoint_and_pinned(
    tmp_path, monkeypatch, forbid_model_imports
):
    rows = _annotations(12)
    pins = {row["question_id"]: ("0" * 64, 1) for row in rows[:10]}
    monkeypatch.setattr(sm, "IMAGE_PINS", pins)
    paths = {qid: tmp_path / f"{qid}.jpg" for qid in pins}
    sizes = {"train": 6, "validation": 2, "test": 2}
    splits = build_sample_dataset(rows, seed=1, sizes=sizes, image_paths=paths)
    assert {k: len(v) for k, v in splits.items()} == sizes and check_split_disjoint(splits)
    assert splits["train"][0]["id"] == "train-0000" and splits["train"][0]["image"].endswith(".jpg")
    assert {r["image_id"] for part in splits.values() for r in part} == set(pins)
    assert build_sample_dataset(rows, seed=1, sizes=sizes, image_paths=paths) == splits
    assert build_sample_dataset(rows, seed=2, sizes=sizes, image_paths=paths) != splits
    with pytest.raises(ValueError, match="split sizes need 11"):
        build_sample_dataset(rows, sizes={"train": 7, "validation": 2, "test": 2})
    with pytest.raises(ValueError, match="absent from the annotations"):
        build_sample_dataset(rows[:5], sizes=sizes)


def test_fetch_sample_dataset_end_to_end_with_injected_fetchers(tmp_path, monkeypatch, forbid_model_imports):
    rows = _annotations(10)
    payloads = {
        row["question_id"]: _picture(tmp_path / f"s{i}.jpg", COLOURS[i % 6]).read_bytes()
        for i, row in enumerate(rows)
    }
    monkeypatch.setitem(CORPUS_FILE, "rows", 10)
    monkeypatch.setitem(CORPUS_FILE, "column_sha256", column_digest(rows))
    monkeypatch.setattr(
        sm, "IMAGE_PINS", {qid: (hashlib.sha256(d).hexdigest(), len(d)) for qid, d in payloads.items()}
    )
    monkeypatch.setattr(sm.time, "sleep", lambda s: None)
    splits = fetch_sample_dataset(
        cache_dir=tmp_path / "cache",
        annotation_fetcher=lambda: rows,
        image_fetcher=lambda qid: payloads[qid],
        sizes={"train": 8, "validation": 1, "test": 1},
    )
    report = validate_dataset(splits["train"])
    assert (
        report["n_records"] == 8
        and report["unique_images"] == 8
        and report["answers_per_question"] == {"min": 10, "max": 10}
    )


# --- dataset validation -------------------------------------------------------------------------------


def test_validate_dataset_reports_and_rejects(tmp_path, forbid_model_imports):
    records = _records(tmp_path)
    report = validate_dataset(records)
    assert report["n_records"] == 12 and report["unique_images"] == 12
    assert report["categories"] == {"unanswerable": 4, "other": 8}
    assert report["digest"] == dataset_digest(report["records"]) and report["model_id"] == MODEL_ID
    assert report["records"][0]["image_size"] == [96, 64]
    good = records
    tiny = _picture(tmp_path / "tiny.jpg", "red", size=(8, 8))
    for bad, message in (
        (good[:7], "8..5000"),
        ([{**good[0], "id": "bad id"}, *good[1:]], "id must match"),
        ([{**good[0], "id": good[1]["id"]}, *good[1:]], "duplicate id"),
        ([{**good[0], "question": ""}, *good[1:]], "non-whitespace"),
        ([{**good[0], "question": "x" * 300}, *good[1:]], "MAX_QUESTION_CHARS"),
        ([{**good[0], "image": str(tmp_path / "missing.jpg")}, *good[1:]], "not found"),
        ([{**good[0], "image": str(tiny)}, *good[1:]], "MIN_IMAGE_SIDE"),
        ([{**good[0], "answers": []}, *good[1:]], "at least 1"),
        ([{**good[0], "answers": ["ok", "  "]}, *good[1:]], "non-empty string"),
        ([{k: v for k, v in good[0].items() if k != "answers"}, *good[1:]], "missing 'answers'"),
        (["not a mapping", *good[1:]], "must be a mapping"),
        ({"a": 1}, "must be a list"),
    ):
        with pytest.raises(ValueError, match=message):
            validate_dataset(bad)
    relative = [{**r, "image": Path(r["image"]).name} for r in records]
    assert validate_dataset(relative, base_dir=tmp_path)["n_records"] == 12


def test_split_dataset_keeps_images_together_and_is_seeded(tmp_path, forbid_model_imports):
    records = _records(tmp_path, n=16)
    records += [
        {**r, "id": r["id"] + "b", "question": "Is it bright?"} for r in records[:6]
    ]  # 2 questions/image
    splits = split_dataset(records, val_fraction=0.15, test_fraction=0.2, seed=3)
    assert sum(len(v) for v in splits.values()) == len(records) and check_split_disjoint(splits)
    assert split_dataset(records, val_fraction=0.15, test_fraction=0.2, seed=3) == splits
    with pytest.raises(ValueError, match="fractions"):
        split_dataset(records, val_fraction=0.5, test_fraction=0.6)
    with pytest.raises(ValueError, match="at least"):
        split_dataset(records, val_fraction=0.0, test_fraction=0.9)
    with pytest.raises(ValueError, match="appears in both"):
        check_split_disjoint({"train": records[:1], "test": [{**records[0], "id": "dup"}]})


# --- metrics and baselines ----------------------------------------------------------------------------


def test_vqa_metrics_and_baselines(tmp_path, forbid_model_imports):
    records = _records(tmp_path)
    golds = [r["answers"] for r in records]
    perfect = vqa_metrics([majority_answer(g) for g in golds], golds)
    assert (
        perfect["vqa_accuracy"] == 1.0 and perfect["majority_match"] == 1.0 and perfect["empty_rate"] == 0.0
    )
    empty = vqa_metrics([""] * len(records), golds)
    assert empty["vqa_accuracy"] == 0.0 and empty["empty_rate"] == 1.0
    with pytest.raises(ValueError, match="gold lists"):
        vqa_metrics(["a"], [["a"], ["b"]])
    assert majority_answer(["Red", "red ", "blue"]) == "red" and majority_answer(["a", "b"]) == "a"
    constant = constant_answer_baseline(records)
    assert constant["majority_match"] == pytest.approx(4 / 12)  # only the unanswerable questions
    assert constant["exact_match"] == 1.0  # every question lists it among its ten answers
    assert 0.0 < constant["vqa_accuracy"] < 1.0 and "unanswerable" in constant["baseline"]
    table = question_prefix_answers(records[:8])
    assert table["what color"] in COLOURS and table["can you"] == "unanswerable" and table[""]
    prefix = question_prefix_baseline(records[:8], records[8:])
    assert prefix["n"] == 4 and 0.0 <= prefix["vqa_accuracy"] <= 1.0 and "prefixes seen" in prefix["baseline"]
    with pytest.raises(ValueError, match="training records"):
        question_prefix_baseline([], records)


# --- BYOD loaders and JSONL ---------------------------------------------------------------------------


def test_byod_json_jsonl_round_trip_and_rejections(tmp_path, forbid_model_imports):
    records = _records(tmp_path, n=8)
    path = write_dataset_jsonl(records, tmp_path / "data.jsonl")
    assert load_byod_dataset(path) == records
    (tmp_path / "data.json").write_text(json.dumps(records), encoding="utf-8")
    assert load_byod_dataset(tmp_path / "data.json") == records
    (tmp_path / "obj.json").write_text('{"records": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="array of records"):
        load_byod_dataset(tmp_path / "obj.json")
    (tmp_path / "data.csv").write_text("id,question\n", encoding="utf-8")
    with pytest.raises(ValueError, match=".json or .jsonl"):
        load_byod_dataset(tmp_path / "data.csv")
    with pytest.raises(FileNotFoundError):
        load_byod_dataset(tmp_path / "missing.json")


# --- adaptation and artifacts without a model ---------------------------------------------------------


def test_adapt_and_artifacts_need_a_loaded_model(tmp_path, forbid_model_imports):
    pipe = _fake_pipeline()
    records = _records(tmp_path)
    with pytest.raises(ValueError, match="epochs"):
        pipe.adapt(records, epochs=0)
    with pytest.raises(ValueError, match="lr"):
        pipe.adapt(records, lr=1.0)
    with pytest.raises(ValueError, match="batch_size"):
        pipe.adapt(records, batch_size=0)
    with pytest.raises(ValueError, match="trainable_decoder_layers"):
        pipe.adapt(records, trainable_decoder_layers=DECODER_LAYERS + 1)
    with pytest.raises(ValueError, match="from_pretrained"):
        pipe.adapt(records)
    with pytest.raises(ValueError, match="call adapt"):
        pipe.save_artifact(tmp_path)
    # evaluate() only needs the answer path, so it works with an injected runner
    metrics = pipe.evaluate(records)
    assert (
        metrics["n"] == 12 and metrics["verdict"] == "measured-small-sample" and metrics["adapted"] is False
    )
    assert (
        metrics["majority_match"] == pytest.approx(4 / 12)
        and metrics["max_new_tokens"] == pl.DEFAULT_MAX_NEW_TOKENS
    )


def test_load_artifact_rejects_bad_manifests_before_touching_weights(tmp_path, forbid_model_imports):
    pipe = _fake_pipeline()
    manifest = {
        "format": ARTIFACT_FORMAT,
        "base_model": {"id": MODEL_ID, "revision": MODEL_REVISION, "weight_sha256": WEIGHT_SHA256},
        "format_version": pl.ARTIFACT_FORMAT_VERSION,
        "files": [{"path": pl.ARTIFACT_WEIGHTS_NAME, "bytes": 1, "sha256": "0" * 64}],
        "tensors": ["text_decoder.cls.predictions.bias"],
        "adapter": {"trainable_decoder_layers": 1},
    }
    (tmp_path / pl.ARTIFACT_MANIFEST_NAME).write_text(json.dumps({**manifest, "format": "other"}))
    with pytest.raises(ValueError, match="artifact format"):
        pipe.load_artifact(tmp_path)
    bad_base = {**manifest, "base_model": {**manifest["base_model"], "weight_sha256": "0" * 64}}
    (tmp_path / pl.ARTIFACT_MANIFEST_NAME).write_text(json.dumps(bad_base))
    with pytest.raises(ValueError, match="different base model"):
        pipe.load_artifact(tmp_path)
    (tmp_path / pl.ARTIFACT_MANIFEST_NAME).write_text(json.dumps(manifest))
    with pytest.raises(FileNotFoundError, match="artifact weights missing"):
        pipe.load_artifact(tmp_path)
    (tmp_path / pl.ARTIFACT_WEIGHTS_NAME).write_bytes(b"x")
    with pytest.raises(ValueError, match="digest or size mismatch"):
        pipe.load_artifact(tmp_path)


def test_load_artifact_refuses_unsupported_versions_extra_files_and_traversal(tmp_path, forbid_model_imports):
    pipe = _fake_pipeline()
    good = {
        "format": ARTIFACT_FORMAT,
        "format_version": pl.ARTIFACT_FORMAT_VERSION,
        "base_model": {"id": MODEL_ID, "revision": MODEL_REVISION, "weight_sha256": WEIGHT_SHA256},
        "files": [{"path": pl.ARTIFACT_WEIGHTS_NAME, "bytes": 1, "sha256": "0" * 64}],
        "tensors": [],
        "adapter": {"trainable_decoder_layers": 1},
    }

    def write(manifest):
        (tmp_path / pl.ARTIFACT_MANIFEST_NAME).write_text(json.dumps(manifest))

    write({**good, "format_version": "0.9"})
    with pytest.raises(ValueError, match="format_version"):
        pipe.load_artifact(tmp_path)
    write({**good, "files": good["files"] * 2})
    with pytest.raises(ValueError, match="exactly one file"):
        pipe.load_artifact(tmp_path)
    write({**good, "files": [{**good["files"][0], "path": "other.safetensors"}]})
    with pytest.raises(ValueError, match="must name exactly"):
        pipe.load_artifact(tmp_path)
    write({**good, "files": [{**good["files"][0], "path": "../" + pl.ARTIFACT_WEIGHTS_NAME}]})
    with pytest.raises(ValueError, match="must name exactly|inside the artifact directory"):
        pipe.load_artifact(tmp_path)
    write({**good, "base_model": {**good["base_model"], "weight_file": "other.bin"}})
    with pytest.raises(ValueError, match="different base weight file"):
        pipe.load_artifact(tmp_path)
    write({**good, "adapter": {}})
    with pytest.raises(ValueError, match="trainable_decoder_layers"):
        pipe.load_artifact(tmp_path)
    write({**good, "adapter": {"trainable_decoder_layers": DECODER_LAYERS + 1}})
    with pytest.raises(ValueError, match="trainable_decoder_layers"):
        pipe.load_artifact(tmp_path)
    write(good)  # every manifest check passes; the weights file is still missing, and no model was imported
    with pytest.raises(FileNotFoundError, match="artifact weights missing"):
        pipe.load_artifact(tmp_path)
