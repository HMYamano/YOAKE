# Release Checklist — YOAKE

Use this checklist before tagging a public release or sharing pre-trained weights. Work through each section in order. Check off each item only after you have personally verified it, not just assumed it is correct.

---

## 1. Pre-Release Code Checks

### Tests and smoke test

- [ ] `python examples/smoke_test.py` passes without errors or warnings
- [ ] `python scripts/train_stage1_detector.py train_anno=data/sample/annotations_train.json val_anno=data/sample/annotations_val.json num_epochs=2` completes without error
- [ ] `python scripts/eval_unified.py anno=data/sample/annotations_val.json checkpoint=outputs/stage4/checkpoint_best.pth` runs and produces a `results_unified.json`
- [ ] All four stage training scripts (`train_stage1_detector.py`, `train_stage2_action.py`, `train_stage3_id.py`, `train_stage4_unified.py`) can be imported without errors

### Code quality

- [ ] No hardcoded absolute paths in any source file under `src/`, `scripts/`, or `tools/`. All paths are relative or configurable via arguments.
- [ ] `import htrtdetr` succeeds in a clean environment after `pip install -e .`
- [ ] All public classes and functions in `src/htrtdetr/` have at minimum a one-line docstring
- [ ] Type hints are present on all function signatures in the main model classes: `HTRTDETRModel`, `HierarchicalTemporalModule`, `MemoryIDHead`, `ActionHead`
- [ ] `pyproject.toml` version field matches the intended release tag (e.g., `version = "0.1.0"`)
- [ ] `LICENSE` file is present at the repository root and contains the correct year and author

### Import and dependency cleanliness

- [ ] `pip install -e .` with only the base dependencies (no `[all]` extras) succeeds and `import htrtdetr` works
- [ ] Optional dependencies (`opencv-python`, `matplotlib`, `wandb`) are not imported at module load time; they are imported lazily with informative `ImportError` messages
- [ ] `requirements.txt` version pins are not overly tight (avoid exact patch pinning where not strictly necessary)

---

## 2. Model Preparation

### Checkpoints present and loadable

- [ ] `weights/stage1_detector.pth` — Stage 1 best checkpoint
- [ ] `weights/stage2_action.pth` — Stage 2 best checkpoint
- [ ] `weights/stage3_id.pth` — Stage 3 best checkpoint
- [ ] `weights/full_model_best.pth` — Stage 4 unified best checkpoint
- [ ] Each checkpoint file is loadable: `torch.load(path, map_location="cpu")` returns a dict with a `model_state_dict` key
- [ ] The config YAML used to produce the released weights is saved to `weights/config.yaml` or referenced in the release notes

### Model card complete (`docs/MODEL_CARD.md`)

- [ ] All metric placeholder dashes (`—`) replaced with real values from `scripts/eval_unified.py`
- [ ] Evaluation metrics in the model card match the output of the evaluation scripts on the test split
- [ ] Training data section describes the actual dataset: video count, total annotated frames, FPS, recording equipment and setup
- [ ] Runtime section reports inference FPS and GPU peak memory measured on representative input (at least 300-frame video)

### Metric verification — run and record each

- [ ] AP50 on held-out test split — `scripts/eval_stage1.py`
- [ ] AP75 on held-out test split — `scripts/eval_unified.py`
- [ ] IDF1 and ID switch count — `scripts/eval_unified.py`
- [ ] Macro action F1 and per-class action F1 — `scripts/eval_unified.py`
- [ ] Inference FPS — `tools/infer_video.py` on a video of at least 300 frames
- [ ] GPU peak memory — `torch.cuda.max_memory_allocated()` or `nvidia-smi dmon`

---

## 3. Documentation Review

- [ ] `README.md` reflects the current command-line interface — no outdated argument names, no references to removed scripts
- [ ] All commands in `README.md`, `docs/TRAINING_GUIDE.md`, and `docs/INFERENCE_GUIDE.md` have been manually tested at least once in a clean environment
- [ ] `docs/DATASET_FORMAT.md` matches the JSON produced by `annotation.py::save_annotation()` — verify field names and types against source code
- [ ] `docs/MODEL_CARD.md` limitations section mentions known failure modes observed during evaluation (e.g., occlusion-induced ID switches, action boundary confusion)
- [ ] `docs/architecture.md` is consistent with the current model implementation in `src/htrtdetr/models/`
- [ ] `docs/assumptions.md` is up to date; any assumption containing "TODO" or a placeholder has been resolved or explicitly deferred with a written rationale

---

## 4. Dataset Licensing Check

- [ ] The training dataset license permits redistribution of model weights derived from it (verify with data owner or legal counsel)
- [ ] If training data is private or restricted, the model card states this explicitly and does not describe the data in identifying detail
- [ ] All third-party components used at training time carry licenses compatible with MIT release:
  - [ ] `torchvision` ResNet-18 ImageNet weights — BSD License
  - [ ] `scipy` Hungarian matching — BSD License
  - [ ] `opencv-python` video I/O — Apache-2.0
- [ ] If any annotation or preprocessing tool was used under a non-permissive license, this is documented in `docs/assumptions.md`

---

## 5. ONNX Export Verification

- [ ] `torch.onnx.export()` completes without errors for the Stage 4 model with a dummy input of shape `(1, 16, 3, 640, 640)`
- [ ] The exported ONNX file passes `onnx.checker.check_model()` validation with no errors
- [ ] ONNX inference output shapes match PyTorch inference output shapes for the same dummy input
- [ ] The ONNX file runs without error under `onnxruntime` or `onnxruntime-gpu`
- [ ] ONNX file size is noted in the release notes (expected: 50–100 MB for ResNet-18 backbone)
- [ ] If dynamic shapes are used, the export has been tested with at least two different input batch sizes

---

## 6. GitHub Release Steps

### Pre-tag checks

- [ ] All changes are committed; `git status` is clean
- [ ] CI passes (if configured): all tests green, `black` and `isort` report no formatting issues
- [ ] Release notes drafted: clear summary of changes since the previous version (or "initial release"), known limitations, and download links for weights

### Create and push the release tag

```bash
git tag -a v0.1.0 -m "YOAKE v0.1.0 — initial public release"
git push origin v0.1.0
```

- [ ] Tag created with the correct semantic version number
- [ ] Tag pushed to the remote repository

### Upload model weights

- [ ] Upload `weights/full_model_best.pth` to the GitHub Release page, Zenodo, HuggingFace Hub, or another permanent file host
- [ ] SHA-256 checksum of the weight file included in the release notes:
  ```bash
  sha256sum weights/full_model_best.pth
  ```
- [ ] Download URL verified to be publicly accessible from an incognito browser session
- [ ] `README.md` updated with the direct download link under a "Pretrained Weights" section

### Update README badges

- [ ] Version badge reflects the new release tag (`v0.1.0`)
- [ ] License badge shows MIT
- [ ] Python version badge is accurate (`>= 3.10`)
- [ ] If a Zenodo or HuggingFace DOI is available, add the DOI badge

---

## 7. Post-Release Monitoring

- [ ] After the release is published, download the weights fresh to a new directory and run `python examples/smoke_test.py` against the downloaded checkpoint to confirm end-to-end reproducibility
- [ ] Check GitHub Issues for any immediately reported bugs, import errors, or broken download links within the first week
- [ ] If a critical bug is found post-release: do not delete the tag or release. Create a patch release (`v0.1.1`) and document the issue and fix clearly in the new release notes
- [ ] Add a "Validated on: YYYY-MM-DD" line to `docs/MODEL_CARD.md` after confirming the released model reproduces the reported metrics in a clean environment
- [ ] If the model is cited in a paper or preprint, update the BibTeX placeholder in `docs/MODEL_CARD.md` and `README.md` with the actual citation once it is available
