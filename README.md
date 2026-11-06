# Fast Speech 2 — English (Svarah) & Telugu (IndicTTS) — Implementation

End-to-end implementation of the [FastSpeech2 assignment](Report/Technical_Report.pdf):
two separate **preprocessing → training → inference** pipelines (one per language), token
durations obtained exclusively from **real forced alignment** (`torchaudio.pipelines.MMS_FA`
+ `uroman`), trained checkpoints, test-sentence audio, variance-adaptor graphs and loss
graphs produced by the trained models — everything trained and run locally on a
6-GB RTX 3050 with mixed precision.

The base repository is [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2);
its original architecture, losses and scheduler are preserved with attribution
(`FastSpeech2_original_README.md`, `requirements_original.txt`). All deviations are
data-handling / training-logistics code and are listed in the section
[Deviations from the base repository](#deviations-from-the-base-repository).

---

## 1. Deliverable layout

```
FastSpeech2/
├── English/          # per-language deliverable wrapper (preprocess / train / inference scripts + config.yaml snapshot)
│   ├── preprocess.py train.py inference.py config.yaml checkpoints/
├── Telugu/           # idem for Telugu
├── Results/          # generated outputs organised by language
│   └── <Language>/
│       ├── test_generated_audios/           # HiFi-GAN-synthesised wavs of the 5 test sentences
│       ├── variance_adaptor_plots/          # per-token predicted duration / pitch / energy plots
│       ├── variance_adaptor_data/           # raw plots for the same predictions
│       ├── training_loss.png …              # training loss graphs (per component)
│       └── validation_*.png                 # validation loss graphs
├── Report/
│   └── Technical_Report.pdf               # technical report of the implementation
├── README.md, requirements.txt
├── config/{en,te}/{preprocess,model,train}.yaml     # actual experiment configurations
├── alignment/{mms_align.py,romanizer.py}, force_align.py, validate_alignment.py
├── preprocess_features.py, train.py, synthesize_samples.py, analyze_variance.py, plot_losses.py
├── preprocessed_data/{English,Telugu}/    # mel/pitch/energy/duration arrays + alignment JSONs + reports
├── processed_data/{English,Telugu}/       # train/val split manifests (basename|speaker|text|raw_text)
├── raw_data/{English,Telugu}/audio/       # source wavs (22.05 kHz preprocessed on the fly)
└── output/ckpt/{en,te}/                   # training checkpoints (step_1000 … step_5000)
```

## 2. Environment

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

Notes:
- `requirements.txt` is our reduced dependency set; the untouched original repository
  list is at `requirements_original.txt`.
- On Windows, set `OPENBLAS_NUM_THREADS=1` before importing torch (`librosa` dependent
  OpenBLAS allocation fails otherwise).
- The two language pipelines are driven by the per-language wrappers, e.g.

```powershell
python English\preprocess.py            # forced alignment -> validation -> feature extraction
python English\train.py                 # 5,000 + 5,000 continuation optimisation steps, resumable checkpoints
python English\inference.py --restore_step 10000
# Same commands with python Telugu\preprocess.py / train.py / inference.py
```

`English\preprocess.py` (`Telugu\preprocess.py`) runs, in order:
1. `force_align.py --language <Language>` — torchaudio `MMS_FA` forced alignment of every
   utterance (resume-safe; per-utterance JSON + `alignment_report.csv`),
2. `validate_alignment.py --language <Language>` — coverage / confidence / duration
   sanity summary written to `preprocessed_data/<Language>/alignment_summary.json`,
3. `preprocess_features.py config/<en|te>/preprocess.yaml --language <Language>` —
   mel / pitch / energy / durations + `stats.json` + `corpus_stats.json`.

## 3. Data

| | **English** | **Telugu** |
|---|---|---|
| Source corpus | [Svarah](https://huggingface.co/datasets/AI4Bharat/Svarah) (AI4Bharat IndicVoices-R), single-speaker sub-corpus | [IndicTTS Telugu](https://www.iitm.ac.in/datasets) (IIIT Hyderabad), single-speaker sub-corpus |
| Raw audio | 16 kHz mono, resampled to 22.05 kHz | 16 kHz mono, resampled to 22.05 kHz |
| Aligned utterances | 3,502 (100%) | 985 / 986 (99.9%) |
| Train / val split | 2,905 / 593 | 820 / 165 |
| Speech hours (train+val) | 5.49 | 5.65 |
| Tokenisation | Unicode grapheme clusters | Unicode grapheme clusters |
| Vocab (tokens incl. `<pad>`, `<unk>`) | 82 | 1,026 |
| Cleaners | `english_cleaners` (number expansion, lowercase) | `telugu_cleaners` (NFC Unicode + whitespace collapse) |

- Splits are **disjoint** utterance lists (`processed_data/<Language>/{train,val}.txt`);
  the split manifest pre-dates preprocessing and is never re-sampled during training.
- `` mean `` grapheme cluster: an utterance's romanization used by the aligner is
  `uroman`; the romanized characters are split back to one span per source grapheme
  cluster (with prefix/word-level fallbacks) so that alignment stays token-level.
- Vocabulary files are `config/en/vocab.json` and `config/te/vocab.json`

## 4. Forced alignment (how durations are obtained)

`alignment/mms_align.py`:

1. load wav via `soundfile`;
2. romanize text with `Uroman().romanize_string`; map each romanized character back to one
   source *grapheme cluster* span (`split_graphemes`),
3. feed ≤ 576 k-sample emission chunks through `torchaudio.pipelines.MMS_FA`
   (`wav2vec 2.0 MMS`, 20-ms frames, 29-symbol CTC emission) and run
   `torchaudio.functional.forced_align` per chunk; spans with an empty end are dropped,
4. token durations = integral frame counts between consecutive cluster boundaries; the
   last token absorbs the rounding remainder so that Σ durations = mel frame count.

Validation results (`preprocessed_data/<Language>/alignment_summary.json`):

| Metric | English | Telugu |
|---|---|---|
| Word coverage | 0.9977 | 1.0000 |
| Mean CTC token score | 0.6979 | 0.9541 |
| Zero-duration-token fraction | 2.3 % | 2.3 % |
| Mean alignment-duration / audio ratio | 0.886 | 0.906 |
| Unknown vocabulary tokens in corpus | 0 / 264,087 | 41 / 116,022 |

Exclusions (documented in `skip_report.json`):
- 4 English utterances have no voiced pitch (`unvoiced`) and cannot provide learnable pitch;
- `telugu_000927`: source text (~330 characters) is far longer than the 8.0 s audio —
  CTC targets do not fit the emission so forced alignment is impossible; the file is
  dropped rather than given synthetic durations.

## 5. Training

Preserved FastSpeech2 architecture (encoder 4×2/256, decoder 6×2/256, FFN 1024, 256 linear
pitch/energy bins) with the base AdamW + Noam schedule. All adjustments are for the
6-GB GPU:

- `mel_batch_budget: 12000` — each batch group is further split greedily so that
  batch size × max-mel-length ≤ 12,000 frames; long recordings (up to 81 s Telugu) are
  kept rather than truncated or dropped;
- `amp: true` — `torch.autocast` + `GradScaler`, gradient clipping applied after `unscale_`;
- loss logging per step into `output/log/<lang>/step_losses.csv` (also visible in
  TensorBoard events under `output/log/<lang>/train`, `output/log/<lang>/val`);
- checkpoint every 1,000 steps to `output/ckpt/<en|te>/` (model + optimizer + LR state),
  so any interrupted run resumes with `--restore_step <step>`.

Measured (RTX 3050 6 GB): ≈ 2.5 GB VRAM with both language trainings running concurrently on one GPU;
English took ≈ 33 min wall-clock for 5,000 steps + 10 validations (≈ 0.40 s/step),
Telugu ≈ 72 min when sharing the GPU with the English run (≈ 0.85 s/step under contention).

## 6. Test sentences & variance-adaptor analysis

The assignment's exact test sentences (E1–E4 including the `10:45 AM, 12 November 2026`
date sentence; T1–T4 including the OCR-broken-in-cluster OCR spellings such as the
split-cluster `చ్చ ింది`, `అడ  విలో`) are preserved verbatim inside
`analyze_variance.py::TEST_SENTENCES` and re-used by `synthesize_samples.py`. For each
sentence the scripts produce:

- `Results/<Language>/test_generated_audios/<id>.wav` — HiFi-GAN-synthesised speech,
- `Results/<Language>/variance_adaptor_plots/{duration,pitch,energy}/<id>.png` —
  per-token predicted value plots with token tick labels,
- `Results/<Language>/variance_adaptor_data/<id>.json` — the plots' numeric data for audit.

`plot_losses.py` derives the training and validation loss figures in `Results/<Language>/`
from the same log files the training run produced.

## 7. Model checkpoints

- English checkpoints: `output/ckpt/en/` (+ copies in `English/checkpoints/`) — 1,000 … 10,000 (5,000 more steps than the planned 5,000 because the first 5,000-step run, evaluated after submission-ready artifacts were regenerated, produced under-converged, quiet audio; the extension resumed from `5000.pth.tar` with `total_step` raised to 10,000 and is fully logged in the same `step_losses.csv`)
- Telugu checkpoints: `output/ckpt/te/` (+ copies in `Telugu/checkpoints/`) — 1,000 … 5,000

Inference uses `--restore_step 10000` for English and `--restore_step 5000` for Telugu.

`Report/Technical_Report.pdf` documents the same experiments (architecture, alignment
validation statistics, loss graphs, variance-adaptor graphs, deviations).

## Deviations from the base repository

- Tokenisation: LJSpeech phoneme pipeline replaced with unicode *grapheme-cluster*
  tokenisation for both languages (`text/grapheme.py`, `build_vocab.py`).
- Telugu romanization + grapheme-cluster span mapping for MMS_FA
  (`alignment/romanizer.py`, `alignment/mms_align.py`).
- `preprocess_features.py`: a new preprocessing entry that builds mel/pitch/energy
  *directly from alignment token boundaries* (the base repo's `prepare_align.py`-based
  MFA path is not used because absolute durations must come from forced alignment).
- `dataset.py`: optional `mel_batch_budget` group-splitting for long-audio VRAM safety.
- `train.py`: AMP training switch, `step_losses.csv` per-step logging;
  `model/optimizer.py`: added `update_learning_rate()` as a public step hook (needed by
  scaler-stepping).
- `text/cleaners.py`: added `telugu_cleaners`; `text/normalization.py`: language dispatch.
- Added build/analysis scripts: `plot_losses.py`, `analyze_variance.py`,
  `synthesize_samples.py`, `build_language_folders.py`, `smoke_test.py`,
  `validate_alignment.py`.
- `prepare_metadata.py`: split-manifest generation fixed to hard-guarantee 1 line = 1
  output row (`|`-separated 4-column format).

## Notes

- `English\preprocess.py` etc. are thin wrappers; the real implementations live at the
  repository root so that the two languages share one verified code path.
- `LICENSE` (MIT) and `FastSpeech2_original_README.md` are kept from the base repository with full attribution.
- **Known limitation — English audio level:** the English model's decoder remains
  text-agnostic at this training budget (its predictions track the corpus-average
  spectrogram; mel-loss plateaus ≈1.26 and validation ≈4.9 while Telugu reaches 0.66
  with the same architecture). The synthesized English test audio is therefore quiet
  and not fully intelligible, and its duration/pitch predictions are flatter than the
  Telugu ones. The training curves showing this behaviour are preserved in
  `Results/English/training_loss.png` and `validation_*.png`.
- ASCII digits/colon are rare in the Telugu corpus, so the date/time tokens of T4 map to
  `<unk>` (41/116,022 tokens = 0.035 % of the corpus).
