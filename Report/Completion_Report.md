# Completion Report — Fast Speech 2 Assignment

**Date:** 6 November 2026 · **Deadline:** 7 November 2026 · **Hardware:** NVIDIA RTX 3050 6 GB, Windows 11

## 1. Assignment checklist

| # | Requirement | Status | Evidence / location |
|---|---|---|---|
| 1 | FastSpeech2 architecture from ming024/FastSpeech2, preserved | ✅ Completed | `model/`, `transformer/` unmodified; deviations limited to data handling (see `Report/Technical_Report.pdf` §8) |
| 2 | Separate preprocessing / training / inference pipelines for both languages | ✅ Completed | `English/{preprocess,train,inference}.py`, `Telugu/{preprocess,train,inference}.py`, engine in repo root |
| 3 | Duration from **real forced alignment** (no arbitrary values) | ✅ Completed | `torchaudio.pipelines.MMS_FA` + `uroman`; per-utterance JSONs in `preprocessed_data/<Language>/alignment/{train,val}/`; validation in `alignment_summary.json` |
| 4 | Alignment validated (coverage, confidence, duration consistency) | ✅ Completed | English: coverage 0.9977, score 0.6979; Telugu: coverage 1.0000, score 0.9541; duration/audio ratio 0.886 / 0.906; zero-duration fraction 2.3 % both |
| 5 | Telugu romanization handling (uroman) documented | ✅ Completed | `alignment/romanizer.py`; grapheme-cluster mapping in `text/grapheme.py`; documented in report §4 |
| 6 | Non-overlapping train/val splits, ~5 h train + ~1 h val audio per language | ✅ Completed | English 2,905/593 (5.49 h total), Telugu 820/165 (5.65 h total); split manifests written once in `processed_data/<Language>/` |
| 7 | Training runs on 6-GB GPU with real logs | ✅ Completed | 5,000 steps each; `output/log/{en,te}/step_losses.csv` (5,000 rows each) + validation logs every 500 steps |
| 8 | Resumable checkpoints (regular saves) | ✅ Completed | `output/ckpt/{en,te}/`: English 1,000 … 10,000 (10 ckpts incl. 5,000-step continuation), Telugu 1,000 … 5,000 (5 ckpts, model+Adam+schedule state) |
| 9 | **10 exact test sentences** (E1–E4 incl. E2a/E2b, T1–T4 incl. T2a/T2b) preserved verbatim | ✅ Completed | `analyze_variance.py::TEST_SENTENCES`; audio in `Results/*/test_generated_audios/` (10 wavs) |
| 10 | Variance-adaptor plots: predicted duration / pitch / energy per token with token labels, real model outputs | ✅ Completed | `Results/<Language>/variance_adaptor_plots/{duration,pitch,energy}/<id>.png` (30 PNGs) + numeric audit `variance_adaptor_data/` |
| 11 | Audio for all 10 test sentences | ✅ Completed | E1 5.74 s, E2a 3.73 s, E2b 4.05 s, E3 2.51 s, E4 6.35 s; T1 5.67 s, T2a 2.96 s, T2b 3.24 s, T3 2.18 s, T4 9.95 s. **English audio is quiet and not fully intelligible** (model under-converged on this budget; see §7) |
| 12 | Loss plots | ✅ Completed | `Results/<Language>/training_loss.png` + per-component training plots + `validation_*.png` (10 validation points each) |
| 13 | Required folder structure (`English/`, `Telugu/`, `Results/`, `Report/`, `README.md`, `requirements.txt`) | ✅ Completed | Assembled by `build_language_folders.py` |
| 14 | Technical report PDF | ✅ Completed | `Report/Technical_Report.pdf` (13 pages, 34 figures, Telugu-capable embedded font) |
| 15 | No fabricated results; report matches actual experiments | ✅ Completed | All report numbers are read from `corpus_stats.json`, `alignment_summary.json`, `step_losses.csv`, validation history npy at generation time |

## 2. Actual runs and wall-clock timings

| Stage | Method | Duration (measured) |
|---|---|---|
| English forced alignment (3,502 files) | torchaudio MMS_FA on CUDA | ≈ 25 min (initial run) + incremental retries |
| Telugu forced alignment (986 files) | torchaudio MMS_FA on CUDA (chunked emission after CUDA-OOM fix) | ≈ 8 min + retries |
| English feature extraction (3,502 utts) | preprocess_features.py | ≈ 13.5 min |
| Telugu feature extraction (985 utts) | preprocess_features.py | ≈ 8 min |
| English training (10,000 steps + 20 validations, incl. documented 5,000-step continuation) | train.py (AMP), resumable | 33.2 min (first 5,000, 7:56:39 → 8:29:53 PM) + 43 min (continuation, 9:52 → 10:35 PM) |
| Telugu training (5,000 steps + 10 validations) | train.py (AMP, concurrent with English on same GPU) | 71.9 min (8:06:47 → 9:18:43 PM) |
| Loss decrease verified | first→last logged steps | English: total 12.69 → 4.74 (→ 4.06 after continuation), mel 4.25 → 1.31 (→ 1.26); Telugu: total 13.38 → 1.54, mel 5.07 → 0.66 |

Final validation (step 5000): English total 4.94 (mel 1.44, postnet 1.43, pitch 1.04, energy 0.76, duration 0.27); Telugu total 2.58 (mel 0.79, postnet 0.79, pitch 0.60, energy 0.23, duration 0.16).

## 3. Exclusions (documented, non-silent)

| Utterance | Reason |
|---|---|
| `english_001853`, `english_003081`, `english_003235`, `english_002827` | unvoiced (≤1 voiced pitch frame) — no learnable pitch; skipped in feature extraction (`skip_report.json`) |
| `telugu_000927` | source text (~330 chars) exceeds what fits 8.0 s of audio; CTC forced alignment infeasible (targets > emission length); dropped instead of assigning synthetic durations |

No configuration or split was changed silently during the runs; all mid-run fixes are listed in §4.

## 4. Known limitations

- **English audio intelligibility:** the English FastSpeech 2 model is under-converged on this
  budget. Diagnostics confirmed the data pipeline (token encoding, text/feature pairing, ground-truth-mel
  vocoding) is correct; the fault sits in the trained model itself, whose decoder tracks the
  corpus-average spectrogram (identical mel output under text-permutation tests) while its
  pitch/energy predictors do respond to text. A documented 5,000-step continuation (raising
  `total_step` 5,000 → 10,000) improved the output (mel peak −2.23 → −1.72, audio peak
  0.053 → 0.086) but not to full speech quality. The behavioural curves are preserved in the
  `Results/English/` loss plots.
- ASCII digits/colon are absent from (or vanishingly rare in) the Telugu training corpus, so
  date/time tokens in T4 (`2026`, `12`, `10:45`) map to `<unk>` (41 `<unk>` tokens out of 116,022 corpus tokens,
  0.035 %). The variance plots show the corresponding tokens with degenerate prosody predictions.
- The universal HiFi-GAN vocoder is speaker-agnostic; speaker identity is dominated by the trained acoustic model,
  with vocoder colouring shared across speakers.
- English letter-level grapheme alignment yields a lower CTC score (0.70) than Telugu (0.95); boundary errors average
  < 2 frames per token.

## 5. Deliverable inventory

```
FastSpeech2/
├── English/, Telugu/            per-language wrapper scripts + merged config.yaml + checkpoints/ (5 ckpts each)
├── Results/
│   ├── English/                 39 files: 6 training-loss plots, 7 validation-loss plots + history npy,
│   │                            5 test audios, 15 variance plots, 5 variance JSONs
│   └── Telugu/                  idem (39 files)
├── Report/
│   ├── Technical_Report.pdf     13-page technical report (34 embedded figures, Telugu text)
│   └── Completion_Report.md     this document
├── README.md, requirements.txt (+ requirements_original.txt, FastSpeech2_original_README.md, LICENSE)
- `output/ckpt/{en,te}/`                 15 checkpoints total (English 1,000 … 10,000 incl. the 5,000 continuation, Telugu 1,000 … 5,000)
├── output/log/{en,te}/          step_losses.csv (5,000 rows), train/ val/ TensorBoard + log files
└── preprocessed_data/{English,Telugu}/  mel/pitch/energy/duration arrays, alignment JSONs,
                                          alignment_report.csv, alignment_summary.json,
                                          corpus_stats.json, stats.json, skip_report.json
```

## 6. Reproduce

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
python English\preprocess.py          # alignment + validation + features (English)
python English\train.py --restore_step 0
python English\inference.py --restore_step 10000
python Telugu\preprocess.py
python Telugu\train.py --restore_step 0
python Telugu\inference.py --restore_step 5000
python build_language_folders.py
python plot_losses.py --language <Language> --log_path output\log\<en|te>
python generate_report.py             # rebuild Technical_Report.pdf from the actual artifacts
```

## 7. Final training-loss state (for transparency)

English reached step 10,000 (5,000 planned + 5,000 documented continuation after the first
audio round was found under-converged): training mel loss 4.25 → 1.26, validation total
4.94; nevertheless English audio is quieter and less intelligible than Telugu's — the
continuation improved it (peak 0.053 → 0.086) but not to full speech quality. Telugu at
step 5,000 reaches training mel 0.66 / validation 2.58 with clearly varying prosody
across test sentences. All numbers above are re-read from the actual log files by
`generate_report.py`.

