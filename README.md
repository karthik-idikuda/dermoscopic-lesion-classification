# Dermoscopic Lesion Analysis

Explainable skin lesion classification on HAM10000: EfficientNet-B3 with Grad-CAM,
quantitative ABCD morphometry, uncertainty quantification and automated severity
grading, served through a FastAPI backend and a browser workstation UI.

> **Not a medical device.** Research and educational prototype only. It has not
> been clinically validated and must never be used to diagnose, treat or rule out
> disease. Any lesion that is new, changing, bleeding or itching needs in-person
> assessment by a clinician regardless of what this software reports.

---

## Status

| Component | State |
|---|---|
| FastAPI backend, 12 endpoints | working |
| Browser UI (analyse, batch, tracking, history, metrics) | working |
| Image quality + out-of-distribution gating | working |
| Hair removal, colour constancy, vignette cropping | working |
| Lesion segmentation + ABCD morphometry | working |
| Grad-CAM / Grad-CAM++ with attention-alignment scoring | working |
| Uncertainty: TTA, MC dropout, BALD | working |
| Severity grading with safety overrides | working |
| Clinical narrative + PDF report | working |
| Longitudinal change tracking | working |
| SQLite case store | working |
| Data-leakage audit | working, reproducible |
| Trained weights (linear probe) | working — 21/21 verification checks |
| End-to-end fine-tune | pending, needs a CUDA GPU |
| Test suite | 149 passing |

### Getting it running from scratch

Nothing large is in version control. Three commands rebuild everything, and
**none of them need a GPU**:

```bash
python scripts/prepare_data.py --skip-segmentations   # 10,015 images -> 195 MB
python scripts/fit_head.py --device cpu               # ~15 min, ~470 MB RAM
python scripts/verify_checkpoint.py models/best_model.pth --images data/ham10000
```

Then `uvicorn app.main:app --reload` and open <http://127.0.0.1:8000>.

For demo images with known ground truth:

```bash
python scripts/make_samples.py --per-class 3   # -> samples/ + INDEX.md
python scripts/demo_samples.py                 # score them all at once
```

### About the weights

The checkpoint is a **linear probe**, and that distinction matters. Full
fine-tuning of EfficientNet-B3 does not fit on an 8 GB laptop — a measured
attempt drove swap to 13.9 GB. So transfer learning was split in two and only the
cheap half runs locally:

1. **Feature extraction, forward only.** Frozen ImageNet backbone under
   `torch.inference_mode()`, batches of 12. Peak RSS 472 MB, no swap growth. All
   10,015 images become 1536-d vectors in ~15 min, cached to disk.
2. **Head fitting.** Only `Linear(1536, 7)` — 10,759 parameters — trained on those
   vectors with class-weighted loss. Seconds on CPU.
3. **Temperature scaling** on the validation split, so displayed confidence is
   calibrated.

The backbone was never fine-tuned, so this is weaker than end-to-end training,
especially on the rare classes. The checkpoint records `training_method` and the
UI and reports label it a probe — never a fine-tuned model.

Always run `verify_checkpoint.py --images` on any checkpoint. Every other check
passes on a permuted head — the tensors load, shapes match, probabilities
normalise — and the result is a model that confidently calls melanoma a nevus.
Only per-class recall on real labelled images catches that.

### Measured results

On the **lesion-grouped** test split (1,523 images, 1,120 lesions, 0% leakage),
via `python -m derm.evaluate`:

| Metric | Value | Reference point |
|---|---|---|
| Accuracy | 54.96% | always-nevus scores 67%, so this is the wrong metric |
| **Balanced accuracy** | **53.69%** | chance 14.3%; always-nevus also 14.3% |
| Macro F1 | 0.379 | every class weighted equally |
| ROC-AUC (macro) | 0.838 | ranking signal is stronger than argmax suggests |
| ECE (calibrated) | 0.096 | from 0.102 before temperature scaling |
| Melanoma recall | 55.5% | argmax only |
| **Melanoma safety-net catch** | **59.2%** | 97/164 escalated to HIGH/CRITICAL |
| Benign over-referral | 15.3% | 191/1,246 — the cost of that safety net |

Per-class precision is the honest weak spot: dermatofibroma 0.065 and vascular
0.139 mean the model over-flags those classes. Recall is even across all seven
(43–67%), which is the class weighting working. Fine-tuning the backbone is where
the remaining gain is.

---

## Quick start

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
# open http://127.0.0.1:8000
```

The UI is served at `/`, interactive API docs at `/docs`.

```bash
pytest                                    # 149 tests, no dataset required
python scripts/smoke_test.py              # end-to-end against a running server
```

---

## Finding: the published 80.17% is inflated

HAM10000 contains **10,015 images of only 7,470 distinct lesions** — 2,545 images
(25.4%) are repeat photographs of a lesion that appears elsewhere in the dataset.
The original notebooks split on *images*, so near-duplicate photographs of the same
physical lesion land in both train and test.

Measured, reproducible, from metadata alone:

```bash
python scripts/audit_leakage.py     # writes docs/split_audit.json
```

| Split | Test images whose lesion is also in train |
|---|---|
| Image-wise (notebook 03, `random_state=42`) | **543 / 1,503 — 36.13%** |
| Lesion-grouped (`derm.data.make_splits`) | **0 / 1,523 — 0.00%** |

Leakage concentrates exactly where it matters most:

| Class | Test images | Leaked | % |
|---|---|---|---|
| Basal cell carcinoma | 77 | 54 | **70.1** |
| Melanoma | 167 | 106 | **63.5** |
| Benign keratosis | 165 | 83 | 50.3 |
| Dermatofibroma | 17 | 7 | 41.2 |
| Actinic keratosis | 49 | 18 | 36.7 |
| Vascular | 22 | 7 | 31.8 |
| Melanocytic nevus | 1,006 | 268 | 26.6 |

So the reported 75% melanoma recall was measured on a test set where 63.5% of the
melanoma images had a twin in training. `derm.data.make_splits` groups on
`lesion_id` by default; expect a lower but honest number. Pass
`--image-wise-split` to reproduce the original behaviour for comparison.

Numbers in `docs/model_comparison.json` are tagged `verified: self_reported` or
`measured`, and the UI labels them accordingly, so a transcribed notebook figure is
never displayed as a measured result.

---

## Layout

```
src/derm/
  config.py         class taxonomy, clinical metadata, all tunables
  preprocessing.py  hair removal (black-hat + Telea), Shades-of-Gray, vignette crop
  quality.py        focus/exposure/glare + skin-chromaticity OOD gate
  segmentation.py   lesion-enhanced Otsu, morphology, centrality-weighted component
  morphology.py     Stolz ABCD + continuous shape/colour descriptors
  model.py          architecture, checkpoint loading, shared inference bundle
  gradcam.py        Grad-CAM / Grad-CAM++ with attention-alignment scoring
  uncertainty.py    TTA, MC dropout, entropy, BALD
  severity.py       composite 0-100 grading with one-directional safety overrides
  report.py         deterministic narrative generator + PDF export
  monitoring.py     longitudinal change tracking (the "E" of ABCDE)
  inference.py      pipeline orchestration
  store.py          SQLite case history
  data.py           dataset discovery, lesion-grouped splitting
  train.py          training CLI
  evaluate.py       evaluation CLI (calibration, safety-net audit)
  baseline.py       SVM baseline (HOG + colour histogram)

app/
  main.py           FastAPI service
  schemas.py        request/response models
  static/           UI — vanilla HTML/CSS/JS, no build step

scripts/
  prepare_data.py       download HAM10000 from Harvard Dataverse (disk-frugal)
  audit_leakage.py      quantify split leakage from metadata alone
  verify_checkpoint.py   validate a checkpoint before trusting it
  smoke_test.py         end-to-end API check
  bench.py              per-device training throughput
```

---

## Pipeline

1. **Quality assessment** on the untouched upload — resolution, focus (variance of
   the Laplacian), exposure, contrast, specular glare, and a skin-chromaticity test
   that rejects images which are not photographs of skin.
2. **Vignette cropping** so the black lens barrel cannot be read as pigment.
3. **Restoration** — directional black-hat hair detection with Telea inpainting,
   then Shades-of-Gray colour constancy. This feeds *geometry only*; the classifier
   receives the un-restored frame, because that is the distribution HAM10000 was
   trained on. Feeding it colour-normalised input would be a silent train/serve skew.
4. **Segmentation** — lesion-enhanced Otsu, morphological cleanup, centrality-weighted
   component selection, falling back to a centred ellipse with reduced confidence.
5. **ABCD morphometry** — asymmetry about the lesion's own principal axes, border
   irregularity across eight sectors, six-colour counting, approximated structures.
6. **Classification** — EfficientNet-B3 averaged over dihedral test-time augmentations.
7. **Uncertainty** — predictive entropy, augmentation disagreement, MC dropout, BALD.
8. **Grad-CAM / Grad-CAM++**, scored against the lesion mask so you can tell whether
   the network attended to the lesion or to an artefact.
9. **Severity grading** — neural risk 52%, morphometry 24%, uncertainty 16%,
   quality 8%, then hard overrides.

### Severity overrides

Overrides can raise a tier but never lower one, because the cost of a missed
melanoma is not symmetric with the cost of an unnecessary referral.

- Melanoma as top class → at least `HIGH` (`CRITICAL` above 70% confidence)
- Melanoma probability ≥ 25% → `HIGH`, even when it is not the top class
- ABCD total dermoscopy score > 5.45 → `HIGH`
- Confidence < 50% → at least `MODERATE`, flagged for review
- Input not skin-like, or no trained weights → `INDETERMINATE`

---

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | liveness and weight status |
| `GET` | `/api/meta` | class taxonomy, tiers, limits |
| `GET` | `/api/metrics` | comparison, evaluation, leakage audit, figures |
| `GET` | `/api/figures/{name}` | serve a figure from `docs/` |
| `POST` | `/api/analyze` | full pipeline on one image |
| `POST` | `/api/analyze/batch` | many images, returned in triage order |
| `POST` | `/api/compare` | longitudinal change between two captures |
| `POST` | `/api/report/pdf` | PDF from a payload or a stored `case_id` |
| `GET` | `/api/cases` | paged case history with filters |
| `GET/PATCH/DELETE` | `/api/cases/{id}` | fetch, annotate, delete |
| `GET` | `/api/cases/stats` | aggregate statistics |
| `POST` | `/api/model/reload` | re-read the checkpoint from disk |

### Security

The API is **unauthenticated by default**, which is only appropriate for local
single-user use on `127.0.0.1`. Set `DERM_API_KEY` to require an `X-API-Key`
header before exposing the port on any network. CORS defaults to localhost
origins; widen with `DERM_CORS_ORIGINS`. Uploads are type- and size-checked
before decoding, and the case store keeps thumbnails rather than full images.

### Configuration

| Variable | Default | Purpose |
|---|---|---|
| `DERM_CHECKPOINT` | `models/best_model.pth` | weights path |
| `DERM_DEVICE` | `auto` | `cuda`, `mps`, `cpu` |
| `DERM_HAM10000_DIR` | — | dataset root |
| `DERM_API_KEY` | — | enable API-key auth |
| `DERM_TTA` | `5` | augmentations per prediction |
| `DERM_MC_PASSES` | `10` | MC dropout passes |

---

## Hardware

Measured on this project's hardware, so you can plan rather than guess:

- **Inference** runs fine on CPU: ~2.9 s per image for the full pipeline
  (classification ~1.6 s, Grad-CAM ~0.9 s) on an Apple M2.
- **Training EfficientNet-B3 needs a real GPU.** On an 8 GB Apple M2 the MPS
  backend drove swap to 13.9 GB and took the boot disk from 9.2 GB to 2.9 GB free
  within ninety seconds. Budget ≥16 GB unified memory, or use CUDA. Use
  `python scripts/bench.py` to measure your own throughput before committing.
- **Dataset**: 2.8 GB of archives. `scripts/prepare_data.py` streams one archive
  at a time and downscales to 256 px on extraction, so peak disk use is ~1.6 GB
  and the final set is ~350 MB instead of ~2.9 GB.

If Python downloads fail with `CERTIFICATE_VERIFY_FAILED: self signed certificate
in certificate chain`, you are behind a TLS-inspecting proxy. `prepare_data.py`
falls back to `curl`, which uses the OS trust store. `pip install truststore` for
a cleaner fix.

---

## Limitations

- HAM10000 is dominated by fair-skinned European and Australian populations.
  Performance on darker skin is uncharacterised and probably worse. This is the
  most serious limitation for real use.
- Only seven categories are modelled. Anything else — squamous cell carcinoma,
  amelanotic melanoma, infections, inflammatory dermatoses — is forced into the
  nearest of the seven and will be wrong.
- The `D` component of ABCD is approximated with classical texture filters rather
  than expert annotation. Treat it as a weak signal. `A` and `B` are faithful to
  the geometric rule; `C` is a colour-quantisation approximation.
- Segmentation is classical, not learned, because HAM10000 ships no masks in the
  main release. It reports its own confidence and falls back to an ellipse.
- Absolute lesion size cannot be recovered from a photograph without a scale
  reference, so change tracking reports size relative to the frame unless you
  supply a field-of-view width.
- The narrative generator is deliberately rule-based, not an LLM: every sentence
  traces to a measurement, and a fluent hallucination cannot be mistaken for a
  clinical finding.

---

## Dataset

Tschandl P., Rosendahl C., Kittler H. *The HAM10000 dataset, a large collection of
multi-source dermatoscopic images of common pigmented skin lesions.* Sci Data 5,
180161 (2018). [doi:10.1038/sdata.2018.161](https://doi.org/10.1038/sdata.2018.161)

Distributed via Harvard Dataverse,
[doi:10.7910/DVN/DBW86T](https://doi.org/10.7910/DVN/DBW86T) (CC BY-NC 4.0).

Method references: Stolz ABCD rule for dermoscopy; Selvaraju et al. Grad-CAM
(2017); Chattopadhyay et al. Grad-CAM++ (2018); Gal & Ghahramani MC dropout (2016);
Houlsby et al. BALD (2011); Guo et al. temperature scaling (2017).
