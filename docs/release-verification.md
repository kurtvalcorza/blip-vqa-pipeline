# Release verification

`tutorials/blip_vqa_colab.ipynb` (`E2E`, **standalone** carrier) is a **release candidate** until the
exact notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation,
code-cell compilation, the generator parity checks and `tools/validate_release_assets.py` are necessary checks but
are **not** runtime evidence under DIMER Notebook Specification 2.0 (REL8). This file is the durable release-gate
record for the notebook.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `E2E` profile, the notebook-spec version
  and the standalone carrier; `metadata.dimer` declares that profile, spec `2.0`, a §3.3 pedagogical mode,
  `standalone: true` and `generated_from` (repository, revision, module SHA-256, generator);
- the standalone carrier (ST1–ST8, PAR1–PAR4): no clone, repository install or repository import on the primary
  path; one cell per carried module (`pipeline.py`, `metrics.py`, `samples.py`), each equal to its source after the
  generator's documented rewrites; the inline `MANIFEST` equal to the committed 8-entry snapshot manifest and the
  inline `PINS` equal to the `pyproject.toml` runtime pins; the notebook byte-identical (on LF) to
  `tools/build_notebook.py` output for its recorded revision; the pinned-install cell with its
  restart-on-stale-import guard; `NOTEBOOK_SOURCE` recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` bound only in the carried module cell (and repeated in the inline manifest, which the
  notebook asserts against the module before fetching), the revision a 40-hex immutable commit, and the same
  identity string in `README.md`, `MODEL_CARD.md` and `docs/WEIGHTS.md` with no stray revisions (the pinned
  VizWiz-VQA dataset revision is the one other 40-hex string allowed);
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`,
  `BlipVQAPipeline.from_pretrained(weights_dir=...)`, `fetch_annotations` and
  `fetch_images` from the pinned cache path, `build_sample_dataset(seed=SPLIT_SEED, image_paths=...)` /
  `load_byod_dataset`, `validate_dataset` per split, `check_split_disjoint`, `write_dataset_jsonl`, the ceiling print,
  `validate_inputs` with the blank-question refusal probe, `pipe.answer` with the sanity checks and the per-image
  `evaluation_report` on the drawn scene, `constant_answer_baseline`, `question_prefix_baseline`, `pipe.evaluate` on
  the frozen model and on the validation and test splits after adaptation with the VQA-accuracy assertion, the
  per-category breakdown, `pipe.adapt` with its explicit hyperparameters, `evaluation_report` on the scene after
  adaptation, `pipe.save_artifact`,
  `BlipVQAPipeline.from_artifact` and the reload-parity assertion, and the provenance fields
  `weight_format`, `weight_sha256` and the `corpus` block), the six expected `outputs/` paths, the learner-facing
  statements (BSD-3-Clause weights, no score exists, not evidence that it describes the image, adaptation with
  accepted answers, the CC BY 4.0 corpus, the unanswerable convention, the two non-neural baselines, majority match,
  no dispersion estimate, the OCR exclusion, the snapshot note) and the
  gated-off BYOD default; forbidden patterns (credential-in-URL, any `git clone` / `github.com` / repository import
  on the primary path, a mutable `revision='main'`, direct `from transformers import` /
  `BlipForQuestionAnswering` / `BlipProcessor` / `.generate(` / `from huggingface_hub import` /
  `get_hf_file_metadata` / `urllib.request` / `pyarrow` / `safetensors` / `torch.optim` / `.backward(` /
  `pipe._model` / `extractall(` use **outside the carried module cells**, `trust_remote_code=True`, `pickle.load`,
  `torch.load(` without `weights_only=True`, `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no document makes an
  unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter (`model_card_spec: "1.1"`), single H1, the 19 required headings in order, and the
  immutable provenance section.

CI also installs the pinned CPU-only torch wheel plus `transformers`, `safetensors`, `numpy`, `pillow`,
`huggingface-hub` and `pyarrow`, runs `ruff check src tests tools`, `tools/build_notebook.py --check`, and the
offline unit suite (`tests/test_pipeline.py`, `tests/test_adaptation.py`, `tests/test_role_helpers.py`,
`tests/test_import_boundary.py`, `tests/test_notebook_parity.py`; injected runner, annotation and image fetchers,
tiny PIL drawings, temporary manifests, no weights — `tests/test_model_backed.py` is skipped without the snapshot). These are
source/provenance and unit checks. They are **not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CPU runtime (CUDA used automatically when present) | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel or equivalent fresh container | Fresh CPU or GPU container, Python 3.12 image; the committed notebook executed verbatim in a fresh interpreter with a `google.colab` shim and **no repository checkout** (the notebook is standalone) | Reproducible clean-room executor of the same class; promotion evidence |
| Local harness (pre-flight only) | Workstation, sequential cell executor with a `google.colab` shim, pre-staged pins | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and **not** promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new CPU or CUDA runtime (Colab, or a fresh-container executor above) with
   **no repository checkout**, an empty Hugging Face cache, and no pre-staged files under the working-directory
   snapshot `weights/blip-vqa-base/` or the corpus cache `weights/vizwiz/` (the standalone path writes the
   manifest itself, stages the missing files from the Hub, reads the pinned VizWiz columns from the Hub and fetches
   the pinned photographs from the VizWiz host, so neither directory may be seeded);
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their defaults:
   `USE_BYOD = False`, `SPLIT_SEED = 42`, `ANSWER_MAX_TOKENS = 10`, `EPOCHS = 4`, `LEARNING_RATE = 5e-5`,
   `BATCH_SIZE = 8`, `TRAINABLE_DECODER_LAYERS = 2`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the revision recorded in
   `metadata.dimer.generated_from` and that the installed core package versions equal the inline `PINS`
   (= `pyproject.toml`): `torch==2.14.0`, `transformers==4.57.6`, `safetensors==0.8.0`, `numpy==2.5.3`,
   `pillow==11.3.0`, `huggingface-hub==0.36.2`, `pyarrow==25.0.1` (an interpreter restart after the install is
   expected where the runtime's preinstalled torch or numpy differ from the pins);
5. verify every default-path stage completes:
   - pinned runtime installed from the inline `PINS` with no GitHub access;
   - the three carried module cells execute (defining `BlipVQAPipeline`, `verify_snapshot`, `stage_missing_files`,
     `validate_inputs`, `evaluation_report`, `anls`, `exact_match`, `vqa_accuracy`, `vqa_metrics`,
     `constant_answer_baseline`, `question_prefix_baseline`, `fetch_annotations`, `fetch_images`,
     `build_sample_dataset`, `validate_dataset`, `check_split_disjoint`, `split_dataset`, `load_byod_dataset`,
     `write_dataset_jsonl`, `gold_texts` and the ceilings) with no import of the repository package;
   - the inline manifest asserted against the module's constants, then `stage_missing_files(WEIGHTS_DIR,
     allow_download=True)` reporting all 8 manifest entries fetched from `Salesforce/blip-vqa-base` at the
     immutable revision on a clean runtime, `verify_snapshot` returning its dict (8 files), and
     `from_pretrained(weights_dir=WEIGHTS_DIR)` loading from the verified directory with `source` `local-snapshot`;
   - Section 4: `fetch_annotations` checking the shard's declared size and SHA-256 against the pins, reading only
     its four text columns (864 rows) and matching the pinned column digest `00b476dd…`; `fetch_images` staging all
     240 pinned photographs from the VizWiz host with every size and SHA-256 matching; the seeded split into
     140 / 40 / 60 questions with `check_split_disjoint` reporting no shared image, the category mix printed and the
     three dataset digests `eca37533…` / `cd46effd…` / `3694b15d…`; `outputs/…_train.jsonl` written; the four
     dataset refusal probes each raising `ValueError`;
   - Section 5: the ceilings (`MIN_IMAGE_SIDE` 16, `MAX_IMAGE_SIDE` 4096, `IMAGE_SIZE` 384, `MAX_QUESTION_CHARS`
     256, `MAX_NEW_TOKENS` 32, `DEFAULT_MAX_NEW_TOKENS` 10, `MIN_RECORDS` 8, `MAX_RECORDS` 5000, `MIN_ANSWERS` 1)
     surfaced; the 640×480 scene drawn; `validate_inputs` writing `outputs/…_input_manifest.json` (verdict
     `accepted`, one recorded rejection finding from the blank-question probe); `pipe.answer` on the seven authored
     questions with every sanity check `True` and the per-image `evaluation_report` verdict `sample-sanity` (the
     card-pass smoke answered five of seven exactly — `2` trees and a `red` door are the recorded misses; a different
     answer on another runtime is a finding to record, not a failure);
   - Section 6: the constant-answer baseline (VQA accuracy ≈ 0.68), the question-prefix baseline (≈ 0.62) and the
     frozen model's test score (VQA accuracy ≈ 0.24, exact match ≈ 0.33, majority match 0.167, ANLS ≈ 0.35
     on CPU float32) with the per-category breakdown (`other` ≈ 0.23, `unanswerable` ≈ 0.08, `yes/no` 1.0 on 5
     questions), and the cell's assertion that the frozen VQA accuracy is above zero;
   - Section 7: `pipe.adapt` printing epoch 0 as the frozen model, 19,526,204 trainable of 361,230,140 parameters,
     140 training photographs, and a four-epoch history with validation VQA accuracy rising (0.167 frozen → 0.683 → 0.783 → 0.758 → 0.775 in the
     build record; `best_epoch` 2);
   - Section 8: `pipe.evaluate` on the validation and test splits with the four-way comparison on four metrics, the
     per-category breakdown and `outputs/…_evaluation_report.json` written (the cell asserts the adapted test VQA
     accuracy exceeds the frozen one — on the sample 0.728 versus ≈ 0.24; majority match 0.167 →
     0.667; `other` ≈ 0.23 → 0.57, `unanswerable` ≈ 0.08 → 0.91);
   - Section 9: the seven scene questions answered by the adapted model with the `sample-sanity` report,
     `outputs/…_answers.csv` written; `pipe.save_artifact` writing `outputs/…_adapter/{adapter.safetensors,manifest.json}`
     (57 tensors, about 78 MB) and `BlipVQAPipeline.from_artifact` reloading it with 8/8 identical answers (the cell
     asserts it); `outputs/…_result.json` written with `NOTEBOOK_SOURCE`, the model identity and licence, the
     snapshot block (`weight_format`, `weight_sha256`), the `corpus` block, the inference-contract items, the
     comparison, the artifact digest, the reload parity, the runtime versions and device;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, Transformers, device), the model
   identifier and immutable revision, whether the model cache, the weights directory and the corpus cache were clean,
   outcome, produced outputs, the observed metrics (as observations, not a benchmark) and any warning or applicable
   `SHOULD` deviation in the tables below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release (REL11).

## Manual clean-runtime evidence

| Notebook | Commit / notebook blob | Date (UTC) | Executor | Outcome |
|---|---|---|---|---|
| `blip_vqa_colab.ipynb` (`E2E`) | generated at `d34d91f` / blob `613d4fa5d4eb` | 2026-09-19 | Local pre-flight harness (Windows, CPython 3.12.10, CPU, `google.colab` shim, pins pre-installed, snapshot, annotations and photographs pre-staged) | PASS — pre-flight only, **not** promotion evidence |
| `blip_vqa_colab.ipynb` (`TASK-INFERENCE`, superseded) | `89fd58c` / `821ff44cd8f5` | 2026-09-14 | Kaggle CPU (`kurtvalcorza/dimer-nb2-blip-vqa` v1) | PASSED — 8/8 code cells, 295.2 s; evidence for the earlier inference-only notebook, not for the `E2E` blob |

## Recorded executions

Notebook identity is the Git blob id of `tutorials/blip_vqa_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/blip_vqa_colab.ipynb`). Wall times, when recorded, are the sum of per-cell times
reported by the executor and include installs and the model download; they are measurements for the stated runtime,
not general estimates.

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-19 | generated at `d34d91f` / blob `613d4fa5d4eb` | Local Windows-venv harness (`run_nb_local.py`: nbclient, fresh `python3` kernel, `CUDA_VISIBLE_DEVICES=-1`, `HF_HUB_OFFLINE=1`, `DIMER_NOTEBOOK_CI_PREINSTALLED=1`), Python 3.12.10, torch 2.14.0+cu130, transformers 4.57.6, snapshot, annotation cache and photographs pre-staged | Default sample path, all 11 code cells: pinned install skipped (pre-installed), `stage_missing_files` reported nothing to fetch, `verify_snapshot` PASS (8 files), annotations read from the pre-staged cache and 240 photographs re-hashed, split 140 / 40 / 60 by image, seven scene answers with the two recorded misses (`sample-sanity` 0.714), baselines VQA accuracy 0.683 / 0.622, frozen test VQA accuracy 0.239 (56.5 s), four epochs 341.8 s (validation VQA accuracy 0.167 → 0.683 → 0.783 → 0.758 → 0.775, epoch 2 kept), adapted test VQA accuracy 0.728 / majority match 0.667 (`other` 0.23 → 0.57, `unanswerable` 0.08 → 0.91), scene unchanged after adaptation (5/7), adapter 78,112,280 B / 57 tensors, reload parity 8/8, 6 outputs written; the committed blob differs from the executed one in markdown prose only (CPU timing estimates filled in after this run) | 616.3 s | PASS — pre-flight only; not promotion evidence |

## Current status

No clean-runtime execution in a **supported** runtime (Colab or Kaggle) has been recorded for the `E2E` blob. What
exists: static validation (`tools/validate_release_assets.py`), the generator parity checks (`--check` OK), the
offline unit suite, the model-backed regressions on CPU and on a local RTX 5070 Ti (CUDA answer, adaptation and
reload), and one **local fresh-kernel execution** of the generated notebook (table above) that exercised the
standalone carrier, the snapshot verification, the column-only annotation fetch, the pinned photographs, the dataset
contract, the inference contract, the bounded fine-tuning, the held-out evaluation, the scene re-answered, the
artifact export and the reload parity — which is necessary but not promotion evidence because the workstation is not a
supported runtime. The registry status remains **Candidate** until a reviewer confirms a recorded supported-runtime
run against the notebook blob under review and an integrator promotes it. Facts a reviewer should weigh: the frozen
model loses to a constant `unanswerable` on the whole VizWiz sample because it never emits that word, so the gain is
read per category and on majority match — a learned `unanswerable` plus a real lift on the answerable questions — and
is several points, not a rescue against a strong baseline; the 40-question validation split makes epoch selection
noisy; the sample's `unanswerable` share is about half, above the benchmark's quarter; and the drawn scene re-answered
after adaptation is one image of evidence about behaviour outside the corpus, not a measurement.
