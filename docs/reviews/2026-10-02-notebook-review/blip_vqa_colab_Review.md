# BLIP VQA-base Visual Question Answering E2E Notebook — Review

**Verdict: Needs revision**  
**Review date:** 2 October 2026  
**Repository:** `kurtvalcorza/blip-vqa-pipeline`  
**Notebook:** `tutorials/blip_vqa_colab.ipynb`  
**Reviewed commit:** `f76032108be338fd43c7224df61c9a3b05e8d845` (`main`, confirmed with `gh api repos/kurtvalcorza/blip-vqa-pipeline/commits/main`)  
**Notebook Git blob:** `82a3c521215fae7b490183ed6c0cce82974ddc89`. This blob was committed at `91f1cf1` (merged as `e7652df`) and executed in the recorded Kaggle run of 2026-09-19. The later commits on `main` touch only the README/model card and the validator/tests (`73cb479`: `tests/test_weight_facts.py`, `tools/validate_release_assets.py`). Generator `--check` exits 0 at the reviewed commit.  
**Finding prefix:** `VQA`

## Executive assessment

On the default path the engineering is careful, and the run reproduces. The notebook does all of the following:

- carries its three modules byte for byte (generator `--check` and `tools/validate_release_assets.py` both exit 0);
- digest-verifies a safetensors-only snapshot;
- reads four text columns of one digest-pinned VizWiz-VQA shard and 240 per-file-pinned photographs;
- splits them by image into 140 / 40 / 60;
- frames the model with two non-neural baselines and four metrics, per category;
- fine-tunes a bounded slice of the answer decoder with validation epoch selection;
- reloads its safetensors adapter with asserted parity.

A direct CPU execution of all 11 code cells, run verbatim with the real weights and the exact pins, reproduced the recorded comparison **to the third decimal**:

| Measure | Value |
|---|---|
| VQA accuracy, constant / prefix / frozen / adapted | 0.683 / 0.622 / 0.239 / 0.728 |
| Majority match, constant / prefix / frozen / adapted | 0.583 / 0.567 / 0.167 / 0.667 |
| Validation VQA accuracy by epoch | 0.167 → 0.683 → 0.783 → 0.758 → 0.775 (epoch 2 kept) |
| Per category (`other` n=33), frozen → adapted, constant | 0.23 → 0.57, constant 0.48 |
| Reload parity | 8/8 |
| Cell time | 253 s |

Six problems stand in the way of `Ready for intended use`:

1. **No one-pass `Run all` (VQA-M1).** The recorded qualification run of this exact blob stopped in cell 3 with the restart `RuntimeError` (`cuda-bindings: loaded=12.9.4, installed=13.4.2; numpy: loaded=2.0.2, installed=2.5.3`). It passed only on a second attempt after a restart, yet the release records call it a PASS.
2. **The headline conclusion is not what the data shows (VQA-M2).** The notebook tells the learner the adaptation "lifts the answerable `other` questions well above both the frozen model and the baselines", and to read majority match first as "the metric a constant answer cannot game". In this review's run the adapted model answered `unanswerable` to **41 of 56** distinct test questions, including **22 of 31** `other` questions. On `other`, 0.43 of its 0.54 VQA accuracy came from those `unanswerable` answers. Its content answers scored **0.108**, *below* the frozen model's 0.204 on the same questions. The constant answer scores 0.583 on majority match.
3. **Result assertions crash honest outcomes (VQA-M3).** Verified: a non-improving run ends Section 8 in a bare `AssertionError`, with no adapter and no `result.json`.
4. **Reruns reuse the adapted model and call it "frozen" (VQA-M4).** Verified with the documented `TRAINABLE_DECODER_LAYERS = 4` experiment.
5. **BYOD contract does not match enforcement (VQA-M5).** The stated minimum is 8 records; the enforced minimum is 50. There are also a stale second upload, subfolder failures and bare `StopIteration`s.
6. **Guided layer missing (VQA-M6).** The notebook is declared `GUIDED`, but the guided layer is largely absent, and 1,652 lines of carried code are unlabelled.

## 1. Review contract and evidence

| Item | Value |
|---|---|
| Declared profile / mode | `E2E` / `GUIDED` (metadata `dimer.notebook_profile` / `notebook_mode`, opening cell) |
| Declared spec | DIMER Notebook Specification **2.0** (metadata, opening cell, `NOTEBOOK_SOURCE`) |
| Spec baseline applied | NOTEBOOK_SPEC **2.2** (2026-09-26), `ml-worker` `origin/main` (last spec commit `50b7eb4`) |
| Intended audience | Not stated. Prerequisites: basic Python and PIL; encoder–decoder generated tokens; VQA accuracy, exact match, majority match and ANLS; "why a confident answer is not a correct one" |
| Supported runtime | "Google Colab or Jupyter, Python 3.12"; CPU float32, CUDA when present; 1.54 GB snapshot plus ~113 MB of photographs |
| Promised outcomes | One-pass `Run all` with no configuration edit; a pinned install; three carried modules; a digest-verified snapshot; digest-pinned VizWiz columns and 240 photographs; a 140/40/60 split by image with four refusals; the inference contract on a drawn scene; two baselines plus the frozen model on four metrics per category; bounded fine-tuning with validation epoch selection; held-out evaluation; the scene re-answered; a safetensors adapter with reload parity; six `outputs/` entries; BYOD "through the same validation, … fine-tuning, held-out evaluation, artifact export and reload-parity cells"; four optional experiments |
| Generator | `tools/build_notebook.py` (`build_notebook.py/2`) + `tools/notebook_template.py`; carried modules from `src/blip_vqa_pipeline/` @ `d34d91f` |

### Evidence actually obtained

- **Source inspection.** All 25 cells (11 code). The carried `pipeline.py` (`adapt`, `evaluate`, `save_artifact`), `metrics.py` (`majority_answer`, both baselines) and `samples.py` (`split_dataset`, `validate_dataset`, `load_byod_dataset`). The generator and template, `README.md`, `STATUS.md`, `MODEL_CARD.md`, `tutorials/README.md` and `docs/release-verification.md`.
- **Documented execution evidence.**
  - Sources: `docs/release-verification.md` and the archived executor summary `.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-blip-vqa/v2/evidence/run_summary.json` (workspace).
  - The run: Kaggle Tesla T4, 2026-09-19, on **blob `82a3c521`, the reviewed blob** (`fetched_blob_verified: true`, clean HF cache).
  - Attempt 1 failed in cell 3 with the restart `RuntimeError` after 163.2 s.
  - Attempt 2 ran 11/11 cells in 287.1 s (`restarted_after_install_cell: true`).
  - No Colab run and no BYOD run of this blob are recorded. `docs/execution-evidence/` does not exist in this repository.
- **Direct execution (this review).**
  - **Environment:** `run_probes.py` on Windows, CPython 3.12, in an existing workspace venv that holds the notebook's exact pins with CPU torch (`torch 2.14.0+cpu`, `transformers 4.57.6`, `safetensors 0.8.0`, `numpy 2.5.3`, `pillow 11.3.0`, `huggingface-hub 0.36.2`, `pyarrow 25.0.1`). CPU only, `HF_HUB_OFFLINE=1`.
  - **Not a clean runtime:** the **real weights** and the pinned VizWiz cache were hard-linked into a scratch working directory. The notebook wrote its own manifest, `stage_missing_files` fetched nothing (`fetched: []`) and every file was re-hashed.
  - **Install skipped:** cell 3 ran with `DIMER_NOTEBOOK_CI_PREINSTALLED=1`.
  - **Execution method:** every code cell ran **verbatim from the notebook JSON** in one namespace. Only form-field literals were substituted, and `google.colab.files.upload` was replaced by a fake.
  - **Stand-in data:** BYOD images are **stand-ins** (Pillow-drawn shapes).
  - **Answer logging:** `BlipVQAPipeline.answer` was wrapped to log each answer. This does not change behaviour.
  - **Probes run:**
    - P1, the default path, 253 s of cell time;
    - P2, the documented four-layer experiment rerun, 70 s;
    - P3, Section 4 BYOD and split structure, with no model;
    - P4, a non-improving run from a fresh base through Section 9, 116 s plus cells 21–23.
  - **Static checks:** JSON parse, compile of all 11 code cells, blob id, generator `--check` (exit 0), `tools/validate_release_assets.py` (exit 0).
- **Not verified:** a Colab run of any kind; a one-pass hosted `Run all`; CUDA in this review; BYOD past Section 4 (Sections 5–9 with uploaded data); the real upload widget; real BYOD photographs; learner understanding.

## 2. Separate judgments

- **Technical correctness:** sound on the default path. Identity, digests, the image-disjoint split, train-only baselines, export and reload parity all hold, and direct execution matched the record exactly. Defects:
  - the restart-dependent install (VQA-M1);
  - result assertions that stop the notebook on legitimate outcomes (VQA-M3);
  - state that survives a rerun and is mislabelled (VQA-M4);
  - the BYOD loader and contract (VQA-M5).
- **Scientific validity:**
  - **Sound:** the protocol, the leakage control and the "one seeded split, no dispersion" caveats.
  - **Not supported:** the central reading. The per-category gain on `other` is the learned `unanswerable` answer, not a lift on answerable content (VQA-M2), and the metric offered as ungameable is gamed by the constant baseline to 0.583.
  - **Minor caveat:** the recipe comparison saw the test split (VQA-m3).
- **Promise fulfilment:**
  - **Delivered:** the default capability list.
  - **Not delivered:** one-pass `Run all` (VQA-M1).
  - **Not deliverable as written:** the optional experiments (VQA-M3, VQA-M4, VQA-m4).
  - **BYOD:** delivered through Section 4 only, for one zip layout of 50+ records uploaded once (VQA-M5).
- **Learner experience:** clear stage prose, two "Look for" notes, honest limits, and three "things to carry to real data". There is no prediction, checkpoint, worked answer, troubleshooting section or conclusion template, and 1,652 carried lines sit unlabelled (VQA-M6). The interpretation guidance the learner *is* given leads to a wrong conclusion (VQA-M2).
- **Spec conformance (2.2):** these applicable `MUST`s fail:

  | Finding | Failed `MUST`s |
  |---|---|
  | VQA-M1 | RUN1, RUN10, ENV6, REL2, REL11 |
  | VQA-M2 | EVAL3 |
  | VQA-M3 | RUN9, UX7 |
  | VQA-M4 | UX7, OUT8, ART8 |
  | VQA-M5 | DAT12, DAT19, REL12 |
  | VQA-m2 | UX12 |
  | VQA-m3 | SPL7, EVAL14 |

  GDL1–GDL15 are largely unmet (SHOULD). The declared spec is 2.0 (VQA-S1).

## 3. Promise and objective tracing

| Claim (where) | Implementation | Observable result (this review) | Learner interpretation |
|---|---|---|---|
| One-pass `Run all`, no intervention (cell 0) | Cell 3: in-kernel `pip install` + stale-import guard | Kaggle record: attempt 1 `RuntimeError` (numpy 2.0.2 → 2.5.3, cuda-bindings 12.9.4 → 13.4.2), restart, attempt 2 ok. Local: install skipped | **Not delivered** (VQA-M1) |
| Pinned, digest-verified snapshot (cells 10–11) | Inline manifest, `stage_missing_files`, `verify_snapshot`, `from_pretrained` | 8 files verified, `fetched: []` (pre-staged), cpu | Delivered |
| Pinned VizWiz columns + 240 photographs; 140/40/60 by image; four refusals (cells 12–13) | `fetch_annotations`, `fetch_images`, `build_sample_dataset`, `validate_dataset`, `check_split_disjoint` | 864 rows, 240 photographs, 140/40/60; four probes rejected with the rule named | Delivered |
| Inference contract on the drawn scene (cells 14–15) | `validate_inputs`, `answer`, `evaluation_report` | 5/7 exact (`2` trees, `red` door as recorded); blank-question probe rejected | Delivered, well explained |
| Baselines + frozen model per category (cells 16–17) | `constant_answer_baseline`, `question_prefix_baseline`, `evaluate`, `by_category` | 0.683 / 0.622 / 0.239; `other` 0.48 / 0.46 / 0.23 | Delivered; "answerable `other`" label wrong (VQA-M2) |
| Bounded fine-tuning with validation selection (cells 18–19) | `adapt(lr=5e-5, epochs=4, layers=2)` | 19,526,204 trainable; val 0.167 → 0.783 (epoch 2); 137 s | Delivered |
| Held-out evaluation; "read majority match first (the metric a constant answer cannot game)"; "the answerable `other` questions … 0.23 → about 0.57" (cells 20–21) | `evaluate` + `assert adapted > frozen` | 0.728; majority match 0.667 vs **constant 0.583**; adapted says `unanswerable` on 22/31 `other`; content answers 0.108 vs frozen 0.204 | **Misread** (VQA-M2); **assertion gates non-improving runs** (VQA-M3) |
| Scene re-answered, export, reload parity (cells 22–23) | `save_artifact`, `from_artifact` | Scene unchanged 5/7; 57 tensors; parity 8/8; 6 outputs | Delivered |
| "lifts the answerable `other` questions well above both the frozen model and the baselines" (cell 24) | — | See row above | **Not supported** (VQA-M2) |
| "set `TRAINABLE_DECODER_LAYERS = 4` and compare …"; "raise `EPOCHS` …" (cell 24) | Rerun of cell 19 | Epoch 0 `'frozen model'` = 0.783 (the previous adaptation; true 0.167); best epoch 0; 38,429,756 trainable reported around the old tensors | **Stale, mislabelled** (VQA-M4) |
| "ask the adapted model `what is written on the sign?` about a photograph with text" (cell 24) | No cell, no photograph, no instruction | — | Not actionable (VQA-m4) |
| BYOD "8..5,000 records", "re-run from that cell" (cells 0, 1, 13) | Cell 13: zip flatten + `load_byod_dataset` + `split_dataset` + per-split `validate_dataset` | 8–49 fail; 50 and 60 pass | Partly delivered (VQA-M4, VQA-M5) |

| Objective (cell 0) | Learner activity | Evidence it was exercised |
|---|---|---|
| Install; read what the carried modules guarantee; stage and verify | Run cells | Procedural only |
| Fetch and validate a pinned corpus; split by image | Read printed digests and refusals | Shown, not practised |
| Answer the drawn scene; read `answer`/`new_tokens`/`truncated` | Read output | Shown; no check of the reading |
| Score the frozen model against two baselines, per category | Read tables | Shown; reading guidance misleads (VQA-M2) |
| Bounded fine-tuning with validation epoch selection; evaluate on a disjoint split | Run cells | Runs on the default (VQA-M3 gates other outcomes) |
| Re-answer the drawing; export and reload with parity | Run cell | Shown |
| (Optional) layers, epochs, sign question, BYOD | Edit a field, no rerun instructions | Reruns stale (VQA-M4) or crash (VQA-M3) |

## 4. Prioritized findings

### VQA-M1 — Major: fresh-runtime `Run all` needs a manual restart after the install cell, yet the release records call it a PASS

- **Cell/section:** Section 1 (cell 3); generator `tools/build_notebook.py:47` (`_INSTALL_GUARD`) and `:471`.
- **Observed issue:** cell 3 pip-installs 9 pins into the running kernel. When a pin replaces an imported distribution, it raises `RuntimeError(… 'Restart the runtime, then rerun from the top.')`.
  - The recorded qualification run of this exact blob did exactly that. `run_summary.json` pass 1 failed after 163.2 s with `cuda-bindings: loaded=12.9.4, installed=13.4.2; numpy: loaded=2.0.2, installed=2.5.3`, and pass 2 ran after a restart.
  - `README.md`, `STATUS.md`, `tutorials/README.md` and `docs/release-verification.md` record "PASSED — 11/11 code cells ok (1 restart after install cell)" and "Release-grade".
  - The procedure (step 4) says the restart "is expected".
  - The opening cell promises a single `Run all` with "no configuration edit".
- **Consequence:** a learner's first `Run all` stops in cell 3. The release status rests on a two-pass run that the spec defines as non-conformant.
- **Evidence:** documented execution (archived executor summary, release record); source inspection. Whether Colab's preloaded packages trigger it identically is **not verified**. The Kaggle image's NumPy 2.0.2 is typical of hosted images, and the pin is 2.5.3.
- **Recommended correction:** adopt the fleet's **uv isolated-environment pattern**, which is how the capstone and newer workshop notebooks already run in one pass.
  - The setup cell bootstraps uv and creates an isolated managed interpreter (`uv venv --managed-python --python 3.12.12 <ROOT>/env`).
  - It installs a hash-locked `requirements.txt` compiled with `uv pip compile` (`uv pip install --require-hashes --only-binary :all:`) and runs the pinned stages in that environment. The kernel's preloaded NumPy/torch are never replaced, so no restart can be required.
  - Reference implementations on `main`: `ast-audio-classification-pipeline/tutorials/DIMER_Sound_Event_Classification_Workshop.ipynb` and `bioclip2-biodiversity-pipeline/tutorials/DIMER_Philippine_Biodiversity_Field_Survey_Capstone.ipynb`.
  - Do not add another in-kernel install guard or loosen pins to dodge the restart.
  - Implement it in the repository's notebook generator (`tools/build_notebook.py` / `tools/notebook_template.py`), regenerate, and re-qualify with a one-pass hosted Run all.
  - Correct the release record so a restart-dependent run is not reported as a `Run all` PASS.
- **Acceptance check:**
  - A fresh Colab (or Kaggle) runtime runs **Run all** once, with no restart and no intervention, through cell 23, and the executor summary shows one pass.
  - README, STATUS, `tutorials/README.md` and `docs/release-verification.md` no longer call the 2026-09-19 two-pass run a Run-all PASS.
  - The procedure no longer calls a restart "expected".
- **Spec:** RUN1, RUN10, ENV6, REL2, REL11 (MUST).

### VQA-M2 — Major: the reading the notebook teaches is not what the data shows; the "answerable `other`" gain is the learned `unanswerable` answer, and majority match is gamed by the constant baseline

- **Cell/section:**
  - Opening cell (majority match "which a constant answer cannot game"; "does … adaptation learn the unanswerable convention *and* improve the answerable questions").
  - Section 6 prose (cell 16, "the answerable `other` questions").
  - Section 8 prose (cell 20: "**majority match** first (the metric a constant answer cannot game)").
  - Interpretation (cell 24: "it lifts the answerable `other` questions well above both the frozen model and the baselines").
  - `docs/release-verification.md` "Facts a reviewer should still weigh" ("a real lift on the answerable questions").
  - Template `tools/notebook_template.py:98`, `:298`, `:372-374`, `:482-483`.
- **Observed issue** (direct execution P1, default path, answers logged; P3, split structure):
  1. **Majority match is gamed.** The constant answer `unanswerable` scores **0.583** majority match on the test split, against 0.667 for the adapted model, because 35 of the 60 test questions have `unanswerable` as their majority answer. Majority match removes VQA accuracy's partial credit, but not a constant answer's advantage on a majority-`unanswerable` sample.
  2. **`other` is not "answerable".** 13 of the 33 `other` test questions have `unanswerable` as their majority answer, and 22 of 33 have it among their accepted answers. That is why the constant answer scores 0.48 there.
  3. **The adapted model mostly answers `unanswerable`.** It did so on **41 of 56** distinct test questions, **22 of 31** distinct `other` questions and 18 of 20 `unanswerable` ones.
  4. **The `other` score is that answer.** On `other`, 0.43 of the adapted model's 0.54 mean VQA accuracy came from those `unanswerable` answers. Its content answers scored **0.108**, against **0.204** for the frozen model on the same questions.
  5. **Conclusion.** The measured gain is the learned convention, applied to most photographs, with no demonstrated lift on answerable content. The adapted model's non-`unanswerable` answers in fact score lower than the frozen model's.
- **Consequence:**
  - **Wrong conclusion.** The learner is taught, in three places, to conclude that bounded adaptation "lifts the answerable `other` questions well above … the baselines". The run does not support that, and the notebook names majority match as the safeguard against exactly this misreading.
  - **Wrong transfer.** A BYOD user carrying the "baselines first" advice to their own data has been shown the wrong check.
  - **Spec reference:** this is a scientific-validity defect (dimension 3) and a result-interpretation defect (dimension 5). EVAL3 requires the principal metrics to be explained by what they measure.
- **Evidence:**
  - Direct execution P1 (CPU, real weights): the comparison is identical to the Kaggle record. The per-answer analysis is keyed by question text, so 4 duplicate-text questions are merged (56 distinct of 60), and that analysis is approximate to that extent.
  - P3 for the split structure (no model).
  - Source inspection of `majority_answer` and the prose.
- **Recommended correction:**
  - Stop calling majority match ungameable. Print the constant baseline's majority match beside the adapted one, as the cell already does, and say what it means.
  - Report, per category, how often each system answers `unanswerable` and the score on questions where it does not (a "content-answer" VQA accuracy). Also report a split by majority answer (majority-`unanswerable` vs answerable), not only by VizWiz's `category` label.
  - Rewrite cells 0, 16, 20 and 24, the README/tutorials README summary and the release-verification "facts" so the claim is what was measured. For example: "the adaptation learns the `unanswerable` convention and applies it to about two thirds of photographs, including many answerable ones; on the questions where it gives a content answer it does not beat the frozen model on this sample".
  - Optionally, make the answerable lift an explicit open question for the learner (GDL10/GDL14).
  - Implement it in the template, regenerate, and record the new numbers.
- **Acceptance check:**
  1. No learner-facing text calls majority match "a metric a constant answer cannot game", or calls `other` "answerable", without qualification.
  2. The notebook prints, per category, the share of `unanswerable` predictions for the frozen and adapted models and the VQA accuracy on non-`unanswerable` predictions.
  3. Any claim of a lift on answerable questions cites a printed number that supports it on the recorded run.
- **Spec:** EVAL3 (MUST); EVAL6, EVAL10, GDL14.

### VQA-M3 — Major: result assertions turn an honest outcome into a crash before export

- **Cell/section:** Section 6 (cell 17: `assert frozen_test['vqa_accuracy'] > 0.0`), Section 8 (cell 21: `assert adapted_test['vqa_accuracy'] > frozen_test['vqa_accuracy']`); template `tools/notebook_template.py:328`, `:409`.
- **Observed issue:** `adapt` keeps epoch 0 whenever no epoch raises validation VQA accuracy (`if current > best_score`). The restored weights are then exactly the starting weights, so the adapted test score *equals* the frozen one and the strict `>` fails.
  - **Direct execution P4:** fresh base, `EPOCHS = 1`, `LEARNING_RATE = 1e-7` (inside the accepted range `(0, 1e-3]`). Validation was 0.167 → 0.167 and `best_epoch` was 0. Cell 21 printed the comparison, wrote `outputs/blip_vqa_evaluation_report.json` and then raised a bare `AssertionError` with adapted = frozen = 0.2389. **No adapter, no reload and no `result.json`** were written.
  - **BYOD (source):** a corpus whose frozen answers match no accepted answer (another language, long free-text answers) stops in Section 6 at `> 0.0`, before fine-tuning.
- **Consequence:**
  - **Who hits it:** a learner whose experiment does not improve validation, or a BYOD user whose 8-plus validation records are not improved by adaptation.
  - **What they see:** a bare `AssertionError` with no explanation.
  - **Spec reference:** RUN9 and UX7 require experiments and BYOD never to block the path.
- **Evidence:** direct execution P4; source inspection of `adapt` and cells 17 and 21.
- **Recommended correction:**
  - Replace both result assertions with reporting. Print the delta, and when the selector keeps epoch 0 or the delta is ≤ 0, say so and explain it. Export still proceeds, and the manifest already records `best_epoch`.
  - Keep `assert` only for invariants such as reload parity.
  - Rewrite cell 20 so learners expect that outcome.
- **Acceptance check:**
  1. With `EPOCHS = 1`, `LEARNING_RATE = 1e-7` from a fresh base, cells 19→23 complete. Cell 21 states that epoch 0 was kept and the delta is 0.000, and `result.json` records it.
  2. `grep -nE "^assert .*(vqa_accuracy)" tutorials/blip_vqa_colab.ipynb` returns nothing.
- **Spec:** RUN9, UX7 (MUST).

### VQA-M4 — Major: rerunning a section after changing a field reuses the adapted model and labels it "frozen"

- **Cell/section:**
  - Section 7 (cell 19), Section 6 (cell 17), optional experiments (cell 24), BYOD instruction (cell 0).
  - Carried `pipeline.adapt`: it starts from the current weights, and `history[0]` is always noted "frozen model".
  - Template `tools/notebook_template.py:41`, `:509`.
- **Observed issue:** `adapt` trains from whatever weights `pipe` holds, and only Section 3 loads a base model. The optional experiments give no rerun instruction.
  - **Section 7 rerun (P2, direct execution).** After the full default run, the documented `TRAINABLE_DECODER_LAYERS = 4` (with `EPOCHS = 1` for time) was applied and cell 19 was rerun.
    - Epoch 0 was printed as `'note': 'frozen model'` with validation VQA accuracy **0.783**, which is the previous adaptation. The true frozen model scores 0.167.
    - The new epoch scored 0.608, so the selector kept "epoch 0".
    - `adapt_result` now reports **38,429,756** trainable parameters and `best_epoch = 0` around tensors trained by the 2-layer run. A Section 8/9 rerun would report and export that description (inferred).
  - **Section 6 rerun after adaptation (source):** it scores the adapted `pipe` and stores it as `frozen_test`.
  - **BYOD "re-run from that cell" (source):** Section 6's "frozen model" is the VizWiz-adapted model, and Section 7 adapts on top of it.
- **Consequence:**
  - The documented experiments report a baseline that is not the base model, under the base model's name.
  - The "compare the per-category scores" experiment cannot be done as written.
  - The artifact's provenance can misdescribe its tensors (OUT8, ART8).
- **Evidence:** direct execution P2; source inspection of `adapt` and `save_artifact`. The Section 6 and BYOD routes are inferred from source.
- **Recommended correction:**
  - Make Section 7 start from the verified base every time. Either reload in the cell, or have `adapt` refuse when `self.adapter is not None` unless told to continue, with a message naming the cell to rerun.
  - Make Section 6 refuse (or reload) when `pipe.adapter is not None`.
  - Record in `history[0]` whether epoch 0 is the base.
  - Give every optional experiment and BYOD a rerun range (for example "rerun from Section 3"), following GDL10.
- **Acceptance check:**
  - After a completed default run, changing `TRAINABLE_DECODER_LAYERS` and rerunning Section 7 either reloads the base (epoch-0 validation VQA accuracy 0.167 on the sample) or stops with an actionable message.
  - Rerunning Section 6 after adaptation either scores the base or stops.
  - On BYOD, Section 6's frozen score is the base model's.
- **Spec:** UX7, OUT8, ART8 (MUST); GDL10, UX10.

### VQA-M5 — Major: the BYOD contract's stated limits and layout are not the enforced ones; bad or repeated uploads fail late, silently or opaquely

- **Cell/section:** Prerequisites (cell 1), BYOD paragraph (cell 0), Section 4 (cell 13); carried `samples.py` `split_dataset` / `validate_dataset`; template `tools/notebook_template.py:41-44`, `:127`, `:160-180`.
- **Observed issue** (direct execution P3, cell 13 verbatim, fake upload, stand-in images):
  1. **Wrong minimum.** The contract says "a dataset needs 8..5,000 records". `split_dataset` takes 20 % for test and 15 % for validation, and cell 13 then validates **each split** with the default `min_records = 8`.
     - 8 records fail with "split leaves 5 training records".
     - 12, 20, 40 and 49 records fail with "2 / 4 / 6 / 7 records; 8..5000 are required". That message names neither the split nor the real rule.
     - The smallest passing dataset is **50** (cell runs and an API scan of 8..80 agree).
  2. **Silent stale data.** `work/byod` is never cleared. Uploading `b.zip` (with `records.json`) after `a.zip` (with `records.jsonl`) printed `data_source: 'BYOD (b.zip)'`, but every training id came from upload A, because `records.jsonl` is searched first.
  3. **Subfolders.** Members are flattened to base names, but `image` is resolved as written. `photos/img_c_000.png` fails with `image file not found: work\byod\photos\img_c_000.png`. Same-named files in different folders would overwrite each other (source).
  4. **Opaque failures.** A cancelled upload and a zip without `records.jsonl` / `records.json` both raise a bare `StopIteration`.
  5. **Provenance and baselines under BYOD.** Cell 13 prints the VizWiz `column_sha256` and `pinned_photographs: 240` under BYOD (observed). From source:
     - cell 23 writes the VizWiz `corpus` block into `result.json`;
     - the constant baseline still answers `unanswerable`, which is meaningful only for corpora with that convention;
     - records without `category` are grouped as `byod`.
- **Consequence:** a user with a modest labelled set meets the documented contract and is rejected with a message about "6 records" after uploading 40. A second attempt trains and evaluates on the wrong data under the new file name.
- **Evidence:** direct execution P3. The empty-answer and missing-image refusals are clear. The downstream BYOD path (Sections 5–9) is **not verified**.
- **Recommended correction:**
  - State the true minimum (derived from the fractions and `MIN_RECORDS`) before upload, or validate the splits with a smaller per-split minimum and name the split in the message.
  - Clear `work/byod` before extracting.
  - Preserve member paths under a guarded root, or reject subfolders with a message.
  - Turn an empty upload or a missing records file into an actionable message.
  - Make the provenance print and `result.json` describe the BYOD source. Let the constant answer be the BYOD training majority answer, or label why it is `unanswerable`.
  - Implement this in `samples.py` and the template, and record a positive and a negative BYOD run that reaches export in `docs/release-verification.md`.
- **Acceptance check:**
  1. The documented minimum equals the smallest set that runs cells 13→23. A smaller set is rejected in Section 4 with a message naming the split and the count required.
  2. A second upload's `data_source` and records are both upload B's.
  3. A zip whose records reference `photos/x.jpg` loads, or is rejected with a layout message.
  4. A cancelled upload prints an actionable message.
  5. Under BYOD, `result.json` carries no VizWiz corpus block.
- **Spec:** DAT12, DAT19, REL12 (MUST); UX10, OUT7.

### VQA-M6 — Major: declared `GUIDED`, but most of the guided layer and any structured learner activity are absent; infrastructure is not labelled

- **Cell/section:** whole notebook; generator `tools/build_notebook.py` (`render`, `:401`) and `tools/notebook_template.py`.
- **Observed issue:**
  - **Absent:** an intended-learner statement, **How to use this notebook**, a roadmap, an Input → Model → Output contract, a glossary, a prediction prompt, an interpretation checkpoint with a worked answer, a troubleshooting section and a conclusion template. The P0 marker counts are 0 for each; the "predict"/"checkpoint" hits are the words "prediction equals" and "checkpoint" meaning the model file.
  - **Objectives are procedural** ("install the pinned runtime; read what the carried … modules guarantee").
  - **Infrastructure is not labelled.** The three carried module cells (834 + 110 + 708 lines) sit between Sections 1 and 3. They are not titled as infrastructure, not collapsed (`cellView: form` count 0) and not marked safe to skip.
  - **Present:** two "Look for" notes, a "Watch" note, honest limits and three things to carry to real data.
- **Consequence:**
  - A self-paced learner must work out alone what to notice in a four-metric, per-category table. Nothing checks their reading, and VQA-M2 shows how easily it goes wrong here.
  - The first screens after the install are 1,652 lines of code that look like prerequisite reading.
- **Evidence:** source inspection; P0 marker counts.
- **Recommended correction:** add the guided layer in the template, following the spec's 2.2 reference notebook (§25.13):
  - audience and prerequisites;
  - how to use the notebook;
  - a roadmap;
  - the Input → Model → Output contract;
  - a glossary: VQA accuracy, majority answer, ANLS, `unanswerable`, epoch selection, adapter;
  - a prediction before Sections 6 and 8 ("how will a constant `unanswerable` score on majority match?");
  - "What to notice" after each stage;
  - collapsible worked answers;
  - one **Predict → Change one thing → Run → Observe → Explain** activity with its rerun range;
  - troubleshooting (install, VizWiz host, BYOD layout);
  - an evidence-based conclusion template.

  Title the carried cells `# @title Infrastructure: …` with `cellView: form`.
- **Acceptance check:**
  - Each of GDL1–GDL14 maps to a named cell.
  - The three carried cells are titled Infrastructure and collapsed.
  - At least one activity asks for a prediction before a result and gives a worked answer after it.
- **Spec:** GDL1–GDL15, UX5, UX8, UX9 (SHOULD).

### VQA-m1 — Minor: doubled braces in the data contract, including the id pattern

- **Cell/section:** cell 0 (BYOD paragraph) and cell 1 (Data contract); template `tools/notebook_template.py:42`, `:127`.
- **Observed issue:** `{{id, image, question, answers}}` and `[A-Za-z0-9_.:-]{{1,64}}` render with doubled braces. The regex as shown is not the enforced `{1,64}`.
- **Consequence:** a user who copies the contract or the pattern gets a wrong schema string and a wrong regex.
- **Evidence:** source inspection; P0 `doubled_braces_in_markdown`.
- **Recommended correction:** use single braces in strings that are not passed through `.format`.
- **Acceptance check:** no `{{` or `}}` in any rendered markdown cell.
- **Spec:** DAT12.

### VQA-m2 — Minor: runtime figures do not name their environment

- **Cell/section:** cell 0 ("about ten minutes of model time"), cell 1 ("the build record measured about 7 s … 57 s … 340 s"), `tutorials/README.md` ("~10 min"); template `:37`, `:125`.
- **Observed issue:** the figures come from the local Windows CPU pre-flight (Intel Core Ultra 9 275HX, per `MODEL_CARD.md`), but the notebook calls them only "the build record", and "about ten minutes" is not labelled as an estimate for a stated machine. For comparison:
  - this review's CPU run took 253 s of cell time;
  - the Kaggle T4 pass 2 took 287 s;
  - a Colab CPU runtime will be slower than either (not measured).
- **Consequence:** a learner on hosted CPU has no basis for the time expectation.
- **Evidence:** source inspection; release record; MODEL_CARD; P1.
- **Recommended correction:** name the environment for each measured figure, and label estimates as estimates.
- **Acceptance check:** every runtime figure in the notebook names its run and machine, or says "estimate".
- **Spec:** UX12 (MUST).

### VQA-m3 — Minor: the default recipe was compared on the test split, while the notebook says the test split was used for nothing else

- **Cell/section:** Section 7 prose (cell 18: "the default is the smallest configuration that captured most of the gain"), Section 8 (cell 20: "never used for training or epoch selection"); `MODEL_CARD.md` §Environment counter-examples (test VQA accuracy 0.728 / 0.722 / 0.756 / 0.761 across four recipes); `tutorials/README.md` ("the test split is used for nothing but the final evaluation"); template `:346`, `:369`.
- **Observed issue:** the recipe comparison that justifies the default reads test-split VQA accuracy, although on an earlier draw with a 30-question validation split. The notebook and the tutorials README describe the test split as untouched.
- **Consequence:** small. The default was kept as "the smallest", not as the best on test, but a learner is not told that the comparison design saw a test split.
- **Evidence:** source inspection.
- **Recommended correction:** say in cell 18/20 and `tutorials/README.md` that the recipe comparison used test-split numbers (of an earlier draw). Preferably compare future recipes on validation only.
- **Acceptance check:** no learner-facing text claims that the test split influenced nothing beyond the final evaluation, unless the documented sweep used only training/validation data.
- **Spec:** SPL6, SPL7, EVAL14.

### VQA-m4 — Minor: the "sign" experiment has no input path, and no experiment states what to rerun

- **Cell/section:** Interpretation, Optional experiments (cell 24); template `:509`.
- **Observed issue:** "ask the adapted model `what is written on the sign?` about a photograph with text" supplies no photograph, no cell and no code pattern. The learner must write `pipe.answer(Image.open(...), ...)` themselves and supply an image. The other three experiments name a field but no rerun range (the consequences are VQA-M4 and VQA-M3).
- **Consequence:** the one experiment aimed at the stated OCR limitation is not practicable for the stated audience.
- **Evidence:** source inspection.
- **Recommended correction:**
  - Ship a small, licence-clean photograph with text (or draw one), with a ready cell.
  - Give each experiment its rerun range.
- **Acceptance check:** each optional experiment names the cells to rerun, and the sign experiment runs from a provided cell without the learner writing code.
- **Spec:** GDL10, UX7.

### VQA-m5 — Minor: the BYOD zip handler has no expanded-size or member-count limit

- **Cell/section:** Section 4 (cell 13); template `:166-171`.
- **Observed issue:** each member is read fully into memory and written with no total-size or count ceiling. Flattening to base names keeps writes inside `work/byod` (no traversal).
- **Consequence:** a large or hostile archive can exhaust disk or memory before validation names a limit.
- **Evidence:** source inspection.
- **Recommended correction:** enforce an expanded-size and member-count limit before writing, and state it in the Prerequisites.
- **Acceptance check:** a zip whose declared uncompressed size exceeds the stated limit is rejected before any member is written.
- **Spec:** §20 (SHOULD), VAL6.

### Suggestions

- **VQA-S1:** update the declared notebook spec from 2.0 to 2.2 (metadata, opening cell, `NOTEBOOK_SOURCE`, References, release docs).
- **VQA-S2:** report a bootstrap interval over the 60 test photographs for the VQA-accuracy and majority-match deltas, and for the per-category deltas (n = 33 / 22 / 5).
- **VQA-S3:** extend reload parity from 8 test photographs to the whole test split (VER4). It costs one more `evaluate` on the reloaded pipeline.
- **VQA-S4:** record per-stage wall times in `result.json` (the device is already recorded), so hosted runs can be compared with the quoted figures.

## 5. Readiness

**Needs revision.** Remaining gates:

1. VQA-M1: a one-pass hosted `Run all` of a revised blob (uv isolated environment), and corrected release records.
2. VQA-M2: interpretation text and printed diagnostics that match what the run measures.
3. VQA-M3 and VQA-M4:
   - no result assertions on held-out metrics;
   - experiments and BYOD that start from the base model, with stated rerun ranges.
4. VQA-M5: a BYOD contract that matches enforcement, recorded with a positive and a negative BYOD run that reaches export (REL12).
5. VQA-M6: the guided layer.

After those, record a fresh clean-runtime run of the new blob in `docs/release-verification.md`.

## 6. Verified versus inferred

- **Verified by direct execution** (CPU, real weights, exact pins, not a clean runtime):
  - the default numbers and parity (P1);
  - the `unanswerable`-dominated adapted answers and the content-answer scores (P1);
  - the stale-rerun mislabel with the documented four-layer experiment (P2);
  - the BYOD minimum, the stale second upload, the subfolder failure, the `StopIteration` cases, the VizWiz provenance print and the test split's majority structure (P3);
  - the `AssertionError` when epoch 0 is kept from a fresh base (P4).
- **Verified from documented evidence:** the restart in the Kaggle qualification run of this blob.
- **Inferred from source:**
  - the Section 6 rerun and BYOD stale-baseline routes;
  - the BYOD `> 0.0` crash;
  - the BYOD `result.json` corpus block;
  - basename collisions and the zip size risk;
  - Colab behaving like Kaggle at the install.
- **Not verified:** Colab, CUDA in this review, BYOD Sections 5–9, real-photograph BYOD, learner understanding.
- **Most likely to be wrong:** VQA-M2's content-answer figures (0.108 adapted vs 0.204 frozen on `other`).
  - They come from answers logged during one CPU run, keyed by question text, which merges 4 duplicate-text test questions (56 of 60 analysed). They are small-sample point estimates.
  - The qualitative finding is robust: 41/56 `unanswerable` answers, and a constant-baseline majority match of 0.583. The exact content-answer gap could shift on CUDA or with a per-record analysis.

Probes: `blip_vqa_colab_Review_Probes.zip` (`run_probes.py`, `results.json`, `source_manifest.json`).
