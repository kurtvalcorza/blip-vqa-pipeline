# BLIP VQA-base visual question answering pipeline

DIMER inference and fine-tuning wrapper for **BLIP fine-tuned on VQA v2** (`Salesforce/blip-vqa-base`), Salesforce's 361M-parameter vision-language model (ViT-B/16 image encoder at 384×384, a BERT-style question encoder fused with the image features, and an answer decoder) that answers a natural-language question about an image with a short generated phrase — pinned to an immutable Hugging Face revision and loaded only from a digest-verified local snapshot. The pipeline accepts one image and one question, decodes greedily under a caller-owned token budget, and returns the answer string with a `truncated` flag; it returns no score and no location. The repository adds a bounded adaptation contract on top: a digest-pinned real VQA corpus (VizWiz-VQA, annotations read column-only and photographs fetched one by one with per-file digests), VQA accuracy, exact match, majority match and ANLS with two non-neural baselines, supervised fine-tuning of the answer decoder's last blocks with validation-VQA-accuracy epoch selection, and a safetensors adapter that reloads onto the digest-verified base.

## Upstream alignment

- Model: `Salesforce/blip-vqa-base`
- Revision: `787b3d35d57e49572baabd22884b3d5a05acf072`
- Upstream weight license: BSD-3-Clause
- Upstream task: visual question answering (VQA v2)
- Repository adaptation: bounded supervised fine-tuning of the answer decoder's last *k* blocks plus its head transform and bias (`adapt`, default 2 of 12 = 19,526,204 of 361,230,140 parameters) on caller-supplied or pinned VizWiz-VQA records; the vision encoder, the question encoder, every embedding and the head's tied projection weight are never modified; the adapter carries only the trained tensors and is bound to the base `model.safetensors` SHA-256

## Quick start

```python
from PIL import Image
from blip_vqa_pipeline import BlipVQAPipeline, exact_match

pipe = BlipVQAPipeline.from_pretrained()        # stages + verifies weights/blip-vqa-base first
result = pipe.answer(Image.open("photo.jpg"), "what color is the house?")
print(result["answer"], result["new_tokens"], result["truncated"])   # generated text; no score exists

# score against accepted answers you know (normalised exact match; vqa_accuracy needs human answer sets)
print(exact_match(result["answer"], ["red", "dark red"]))
```

Adaptation on the pinned VizWiz-VQA sample (CPU, about six minutes of training after the snapshot and the 240 photographs are staged):

```python
from blip_vqa_pipeline import (
    BlipVQAPipeline, fetch_sample_dataset, check_split_disjoint, constant_answer_baseline, question_prefix_baseline,
)

splits = fetch_sample_dataset()            # annotations column-only from one pinned Hub shard + 240 pinned photographs (~113 MB), cached under weights/vizwiz/
check_split_disjoint(splits)               # 140 / 40 / 60 questions, one photograph each, no image shared between splits
pipe = BlipVQAPipeline.from_pretrained()
print(constant_answer_baseline(splits['test'])['vqa_accuracy'], pipe.evaluate(splits['test'])['vqa_accuracy'])   # 0.683, 0.239 in the recorded run
pipe.adapt(splits['train'], splits['validation'])                  # answer decoder's last 2 blocks + head, 4 epochs, best validation VQA accuracy kept
print(pipe.evaluate(splits['test'])['vqa_accuracy'])               # 0.728 in the recorded run (per category in the notebook)
artifact = pipe.save_artifact('outputs/adapter')                   # adapter.safetensors (~78 MB) + manifest.json
again = BlipVQAPipeline.from_artifact(artifact)                    # verifies base digest + artifact digest before applying
```

Install into a Python 3.12 environment that already holds the pinned dependencies with `pip install -e . --no-deps`; run `pytest -q -o addopts= tests` for the offline test suite (no weights needed). On a fresh clone the manifest is committed but the weights are not: `BlipVQAPipeline.from_pretrained(allow_download=True)` fetches exactly the missing manifest-listed files at the pinned revision, then verifies them.

`answer(image, question, *, max_new_tokens=10)` takes a PIL image with sides in 16..4,096 px, a non-empty question of at most 256 characters and a budget in 1..32, and returns `answer`, `question`, `image_size`, `new_tokens`, `truncated`, the generation settings, `device`, `source`, `model_id` and `model_revision`. `evaluate(records)` answers a validated dataset and reports mean VQA accuracy, the exact-match and majority-match rates, mean ANLS and the empty rate (`measured` / `measured-small-sample`); `adapt(train, val, *, epochs=4, lr=5e-5, batch_size=8, trainable_decoder_layers=2, seed=0)` fine-tunes the answer decoder's last blocks and head on each record's majority answer, with the frozen vision features computed once per training image, and keeps the best-validation-VQA-accuracy epoch; `save_artifact` / `from_artifact` export and reload the trained tensors as safetensors with a manifest bound to the base weight digest. Dataset helpers (`fetch_annotations`, `fetch_images`, `build_sample_dataset`, `fetch_sample_dataset`, `validate_dataset`, `split_dataset`, `check_split_disjoint`, `load_byod_dataset`, `write_dataset_jsonl`) live in `samples.py`; `vqa_metrics` and the baselines (`constant_answer_baseline`, `question_prefix_baseline`) in `metrics.py`; records are 8–5,000 mappings `{id, image, question, answers}` whose image decodes and whose answers are non-empty, and every ceiling is a refusal, never a silent cut.

## Weights layout

```
weights/blip-vqa-base/
  dimer-base-manifest.json   # modelId, revision, per-file bytes + SHA-256 (8 files)
  config.json                # BlipForQuestionAnswering: ViT-B/16 @ 384 + 12-layer text encoder/decoder (vocab 30524)
  preprocessor_config.json   # BlipImageProcessor: resize 384x384, CLIP mean/std
  special_tokens_map.json  tokenizer.json  tokenizer_config.json  vocab.txt
  model.safetensors          # git-ignored, 1,538,800,584 bytes
  README.md
weights/vizwiz/              # git-ignored: annotations.json (the pinned shard's text columns) + images/ (240 pinned photographs, ~113 MB), digest-checked on read
```

`pytorch_model.bin` and `tf_model.h5` exist upstream and are deliberately not listed (pickle / TensorFlow port; the executed artifact is the SafeTensors file).

## Input ceilings and request parameters

`MIN_IMAGE_SIDE = 16`, `MAX_IMAGE_SIDE = 4096`; `MAX_QUESTION_CHARS = 256` (one non-empty question per call, whitespace collapsed); `MAX_NEW_TOKENS = 32`, `DEFAULT_MAX_NEW_TOKENS = 10`; `DECODING = "greedy"`; `IMAGE_SIZE = 384` (the processor's fixed resize, aspect ratio not preserved; documentation only); datasets of `MIN_RECORDS = 8`..`MAX_RECORDS = 5000` records with at least `MIN_ANSWERS = 1` accepted answer each. See `MODEL_CARD.md` for who owns the budget, why four metrics are reported per category, and the measured CPU timings.

## Tests

```
pip install -e . --no-deps
pytest -q -o addopts= tests
```

Tests are offline: they use an injected fake runner, injected annotation and image fetchers, tiny PIL drawings and temporary manifests, never the weights (35 tests plus 5 notebook-parity tests). `tests/test_model_backed.py` (6 tests: `evaluate` against accepted answers, a one-epoch adaptation of the last decoder block with an artifact round trip, the final-epoch policy, the loader's scope check, the transactional guarantee, and — where CUDA is visible — answer, adaptation and reload on the accelerator) runs only when `weights/blip-vqa-base/` is staged.

## Tutorial

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/blip-vqa-pipeline/blob/main/tutorials/blip_vqa_colab.ipynb)

`tutorials/blip_vqa_colab.ipynb` is declared `E2E` (mode `GUIDED`) under DIMER Notebook Specification 2.0 and is **standalone** (§4): generated by `tools/build_notebook.py`, it carries the three pipeline modules, model identity, manifest digests and runtime pins, so the exported notebook runs without this repository (parity enforced by `tests/test_notebook_parity.py`; see `tutorials/README.md`). Its default path reads the four text columns of one digest-pinned VizWiz-VQA shard from the Hub (about 0.13 MB, CC BY 4.0, refused on any size, file-digest or column-digest mismatch), fetches the 240 pinned photographs from the VizWiz host (about 113 MB, each refused on any mismatch), cuts them by image into 140 / 40 / 60 records with four refusal probes, stages the git-ignored `model.safetensors` with `stage_missing_files(..., allow_download=True)` and digest-verifies the snapshot, exercises the inference contract with its input manifest and sanity checks on a drawn scene, scores the constant-answer and question-prefix baselines and the frozen model on the test photographs (VQA accuracy 0.683 / 0.622 / 0.239 in the recorded run — the frozen model never says `unanswerable`), fine-tunes the answer decoder's last two blocks and head for four epochs with validation-VQA-accuracy epoch selection (341.8 s on CPU), re-scores the test photographs (VQA accuracy 0.728, majority match 0.167 → 0.667) with a per-category breakdown, re-answers the drawn scene with the adapted model, exports a ~78 MB safetensors adapter and reloads it with 8/8 identical answers. Every number is one seeded split with no dispersion estimate. BYOD (one zip of images plus a `records.jsonl` / `records.json` of `{id, image, question, answers}`) is optional and gated off by default. See `docs/release-verification.md` for the release gate.

## Release status

**Release-grade** — the `E2E` notebook blob `82a3c521` (committed at `91f1cf1`) executed top-to-bottom in a clean Kaggle Tesla T4 runtime on 2026-09-19 (11/11 ok (1 restart after install cell), 450.3 s); the record is in `docs/release-verification.md` and `STATUS.md`. Static and unit checks — including the standalone generator parity checks — are necessary but were never the evidence; the hosted run is. A later change to the carried modules or the notebook returns the status to Candidate until re-verified.

## Documentation

- `MODEL_CARD.md` — MODEL_CARD_SPEC 1.1 card, provenance digests, input/output and adaptation contract, measured runtime.
- `docs/WEIGHTS.md` — weight provenance, adapter and corpus notes.
- `STATUS.md` — release status.

## Licensing

This repository's code is Apache-2.0 (see `LICENSE`). The upstream weights are BSD-3-Clause; the tutorial corpus is CC BY 4.0 and is not redistributed; see `docs/WEIGHTS.md` and `MODEL_CARD.md`.

## AI Assistance Disclosure

This repository’s code and accompanying documentation were developed with generative AI assistance for code development and technical writing under maintainer direction. The maintainer remains responsible for reviewing the implementation, validating results, and making release decisions. AI assistance does not constitute independent verification, provider endorsement, or release approval.
