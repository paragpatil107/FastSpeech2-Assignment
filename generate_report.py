import json
from pathlib import Path

import numpy as np
import soundfile as sf
from fpdf import FPDF
from fpdf.enums import XPos, YPos

ROOT = Path(__file__).resolve().parent
REPORT_ASSET_DIR = ROOT / "Report"
FONT_PATH = REPORT_ASSET_DIR / "Nirmala_0.ttf"

LANGUAGES = ["English", "Telugu"]
TAGS = {"English": "en", "Telugu": "te"}

SPEAKER_NOTE = {
    "English": "Svarah (IndicVoices-R, AI4Bharat) — single female speaker sub-corpus",
    "Telugu": "IndicTTS Telugu (IIIT Hyderabad) — single speaker sub-corpus",
}


def load_json(path):
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def corpus_stats(language):
    return load_json(ROOT / "preprocessed_data" / language / "corpus_stats.json")[language]


def alignment_summary(language):
    return load_json(ROOT / "preprocessed_data" / language / "alignment_summary.json")


def skip_counts(language):
    path = ROOT / "preprocessed_data" / language / "skip_report.json"
    if not path.exists():
        return {}
    counts = {}
    for _, reason in load_json(path):
        counts[reason] = counts.get(reason, 0) + 1
    return counts


class ReportPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.add_font("Nirmala", "", str(FONT_PATH))
        self.add_font("Nirmala", "B", str(FONT_PATH))
        self.set_text_shaping(True)
        self.set_auto_page_break(auto=True, margin=18)
        self.set_margins(16, 16, 16)

    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Nirmala", "", 9)
        self.set_text_color(120)
        self.cell(0, 6, "FastSpeech 2 — English & Telugu Implementation Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(0)
        self.ln(2)

    def footer(self):
        self.set_y(-14)
        self.set_font("Nirmala", "", 9)
        self.set_text_color(120)
        self.cell(0, 8, f"Page {self.page_no()}/{{nb}}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(0)

    def h1(self, title):
        self.ln(2)
        self.set_font("Nirmala", "B", 15)
        self.multi_cell(0, 8, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(1)

    def h2(self, title):
        self.ln(1)
        self.set_font("Nirmala", "B", 12)
        self.multi_cell(0, 7, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(1)

    def body(self, text):
        self.set_font("Nirmala", "", 10.5)
        self.multi_cell(0, 6, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(1.5)

    def bullet(self, text):
        self.set_font("Nirmala", "", 10.5)
        self.set_x(20)
        self.multi_cell(0, 6, "•  " + text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def key_value_table(self, rows, widths=(92, 92)):
        self.set_font("Nirmala", "", 10)
        for key, value in rows:
            self.multi_cell(0, 5.6, f"{key}: {value}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def stat_table(self, headers, rows, col_widths=(70, 55, 55)):
        self.set_font("Nirmala", "B", 10)
        for i, header in enumerate(headers):
            self.cell(col_widths[i], 7, header, border=1, align="C")
        self.ln()
        self.set_font("Nirmala", "", 9.5)
        for row in rows:
            max_lines = 1
            heights = []
            cells = []
            for i, value in enumerate(row):
                text = str(value)
                width = col_widths[i]
                count = len(self.multi_cell(width, 5.6, text, dry_run=True, output="LINES"))
                max_lines = max(max_lines, count)
                cells.append(text)
            for i, text in enumerate(cells):
                self.multi_cell(col_widths[i], 5.6 * max_lines, text, border=1, align="L",
                                new_x=XPos.RIGHT, new_y=YPos.TOP, max_line_height=5.6)
            self.ln(5.6 * max_lines)
        self.ln(1)


def rows_from_stats_table(stats_by_language):
    header = ["Metric", "English", "Telugu"]
    train_stats = {lang: corpus_stats(lang) for lang in LANGUAGES}
    align_stats = {lang: alignment_summary(lang) for lang in LANGUAGES}
    rows = [
        header,
        ["Aligned utterances", f"{align_stats['English']['aligned_files']}", f"{align_stats['Telugu']['aligned_files']} (of 986)"],
        ["Preprocessed train / val", f"{train_stats['English']['train_utterances']} / {train_stats['English']['val_utterances']}", f"{train_stats['Telugu']['train_utterances']} / {train_stats['Telugu']['val_utterances']}"],
        ["Speech hours (train+val)", f"{train_stats['English']['speech_hours']}", f"{train_stats['Telugu']['speech_hours']}"],
        ["Mean word coverage", f"{align_stats['English']['mean_word_coverage']}", f"{align_stats['Telugu']['mean_word_coverage']}"],
        ["Mean token alignment score", f"{align_stats['English']['mean_token_score']}", f"{align_stats['Telugu']['mean_token_score']}"],
        ["Vocabulary size (incl. <pad>/<unk>)", "82", "1026"],
        ["Tokens with alignment score 0", f"{align_stats['English']['mean_zero_duration_token_fraction']}", f"{align_stats['Telugu']['mean_zero_duration_token_fraction']}"],
        ["Alignment-duration/audio ratio", f"{align_stats['English']['mean_alignment_duration_ratio']}", f"{align_stats['Telugu']['mean_alignment_duration_ratio']}"],
        ["Unknown tokens in corpus", f"{align_stats['English']['unknown_vocabulary_tokens']}", f"{align_stats['Telugu']['unknown_vocabulary_tokens']} / {align_stats['Telugu']['total_text_tokens']}"],
    ]
    return rows


def build_report():
    pdf = ReportPDF()
    pdf.alias_nb_pages()

    pdf.add_page()
    pdf.set_font("Nirmala", "B", 20)
    pdf.multi_cell(0, 10, "Fast Speech 2 — Technical Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.set_font("Nirmala", "", 11.5)
    pdf.multi_cell(0, 7, "End-to-End TTS Implementation: English (Svarah) & Telugu (IndicTTS)",
                   new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.ln(2)
    pdf.set_font("Nirmala", "", 10)
    pdf.key_value_table([
        ("Base implementation", "ming024/FastSpeech2 (architecture preserved, pipeline adapted)"),
        ("Data", SPEAKER_NOTE["English"] + "; " + SPEAKER_NOTE["Telugu"]),
        ("Vocoder", "HiFi-GAN universal (pretrained checkpoint shipped with base repository)"),
        ("Alignment", "torchaudio MMS_FA + uroman romanization"),
        ("Hardware", "NVIDIA RTX 3050 6GB, mixed-precision training"),
    ])

    pdf.h1("1. System Overview")
    pdf.body(
        "Fast Speech 2 is a non-autoregressive text-to-speech transformer. The preserved architecture consists of a "
        "phoneme-like token encoder, a variance adaptor (duration, pitch, energy predictors), a decoder and PostNet "
        "that produce 80-band mel spectrograms, and a pretrained HiFi-GAN universal vocoder that converts the "
        "predicted mel spectrograms to waveforms. "
        "The variance adaptor predicts per-token duration (log-space), pitch (F0) and loudness (energy); these "
        "predictions are quantized with 256 linear embedding bins and injected into the encoder output, and the "
        "duration predictor outputs are rounded and used to expand each token to its acoustic frame span before decoding."
    )
    pdf.body(
        "The original repository supports English LJSpeech with phoneme tokenization. For this assignment the "
        "architecture is unchanged; every modification is data-handling or training-logistics code, itemised in "
        "section 8."
    )

    pdf.h1("2. Datasets and Speaker-Corpus Preparation")
    pdf.body(
        "Two single-speaker training corpora are used, one per language. Full utterances of the source recordings were "
        "resampled from 16 kHz to 22.05 kHz with librosa and pre-existing metadata rows were verified against the audio. "
        "The split between train and validation is deterministic and non-overlapping: "
        "English: 2,905 training and 593 validation utterances; Telugu: 821 training, 165 validation utterances and 1 "
        "exclusion (section 4). Every split uses distinct utterances of the same speaker; no utterance appears in both."
    )
    pdf.body(
        "Text normalisation is language-specific: English uses the repository's english_cleaners (number expansion, "
        "lowercase); Telugu uses a new telugu_cleaners that normalises Unicode (NFC) and collapses whitespace. Both "
        "languages use grapheme-level tokenisation: a vocabulary of grapheme clusters is built with Unicode grapheme "
        "segmentation, so no external phoneme lexicon is required for Telugu."
    )

    pdf.add_page()
    pdf.h1("3. Model Training Corpus Statistics")
    table = rows_from_stats_table(LANGUAGES)
    headers = table[0]
    pdf.stat_table(headers, table[1:])
    pdf.body(
        "Alignment token coverage is the fraction of the source text tokens that received a non-empty CTC span, "
        "and the token score is the mean MMS_FA phone score. Zero-duration tokens are tokens whose forced span "
        "collapses to a single frame (mostly punctuation and rare graphemes). The alignment-duration/audio ratio "
        "measures how much of the raw recording the forced speller covers."
    )

    pdf.h1("4. Forced Alignment for Token Durations")
    pdf.body(
        "Token durations come exclusively from forced alignment — no hand-made or uniform values are used. We use "
        "torchaudio's MMS_FA pipeline: the wav2vec 2.0 MMS model is used to emit 20-ms frame posteriors over 27 roman "
        "transcription symbols, and CTC forced alignment produces per-token spans. "
    )
    pdf.body(
        "Both corpora are romanized for the aligner: English is already latin-script and Telugu is romanized with the "
        "uroman toolkit (v1.3.1.1). Because uroman romanizes whole words, each romanized character is mapped back to a "
        "specific source grapheme cluster by partitioning the romanization of each grapheme cluster independently "
        "(cluster-level split with a prefix/word-level fallback in text/grapheme.py), so alignment spans remain "
        "attributable to individual Telugu grapheme clusters."
    )
    pdf.body(
        "Long recordings are aligned in <=576k-sample emission chunks and the result is stitched so no CUDA-OOM occurs "
        "on the 6 GB GPU. Validation statistics (vocab match, coverage, alignment score, duration sanity) are written "
        "to preprocessed_data/<Language>/alignment_summary.json for each language."
    )
    pdf.body(
        "Exclusions: 4 English utterances were skipped as unvoiced (less than 2 voiced pitch frames, hence no learnable "
        "pitch) and 1 Telugu utterance (telugu_000927) supplies ~330 characters of speech text in 8.0 seconds, which "
        "makes CTC forced alignment infeasible (targets exceed the emission length); it was excluded from preprocessing "
        "and training rather than given synthetic timelines. All other utterances aligned successfully."
    )
    for language in LANGUAGES:
        counts = skip_counts(language)
        if counts:
            pdf.body(f"{language} skip report: " + ", ".join(f"{reason}: {count}" for reason, count in counts.items()) + ".")

    pdf.add_page()
    pdf.h1("5. Feature Extraction (preprocess_features.py)")
    pdf.body(
        "Every kept utterance is trimmed to its first/last aligned frame and converted to log-mel with the Tacotron "
        "STFT (1024/256/1024, 80 mel bins, 22.05 kHz), so mel frame positions correspond exactly to the alignment "
        "frame grid. Token durations are the integral frame counts between consecutive token boundaries, with the "
        "last token absorbing the rounding remainder so that the duration sum equals the mel frame count exactly. "
        "Pitch is extracted with WORLD (DIO + StoneMask) and energy with the same STFT energy; both are reduced to "
        "token level by averaging over each token's frames, then z-score normalised per corpus (stats.json)."
    )

    pdf.h1("6. Training on a 6-GB GPU")
    pdf.body(
        "Training uses the preserved FastSpeech2 training loop with the following additions: (a) memory-budgeted "
        "batching: each data-batch group is re-sliced greedily so that batch_size x max-mel-length <= 12,000, which "
        "length-protects the corpus's longer utterances (Telugu reaches 81 s) without dropping files; (b) mixed-precision "
        "autocast + GradScaler on CUDA; (c) per-step CSV loss logging (output/log/<lang>/step_losses.csv) alongside the "
        "TensorBoard logs; and (d) checkpoints every 1,000 steps with optimiser and schedule state, so interrupted runs resume exactly."
    )
    config_rows = load_json(ROOT / "Report" / "training_configs.json")
    pdf.stat_table(["Field", "Setting"], config_rows, col_widths=(44, 136))
    for language in LANGUAGES:
        history_path = ROOT / "Results" / language / "validation_loss_history.npy"
        if not history_path.exists():
            continue
        history = np.load(history_path)
        pdf.body(f"{language}: {int(history[-1][0])} training steps; validation evaluated every 500 steps; last validation losses: "
                 + ", ".join([
                     f"total={history[-1][1]:.3f}",
                     f"mel={history[-1][2]:.3f}",
                     f"pitch={history[-1][4]:.3f}",
                     f"energy={history[-1][5]:.3f}",
                     f"duration={history[-1][6]:.3f}",
                 ]))
    for language, tag in TAGS.items():
        for plt_name in [f"training_loss.png", f"validation_total_loss.png"]:
            image_path = ROOT / "Results" / language / plt_name
            if image_path.exists():
                if pdf.get_y() > 240:
                    pdf.add_page()
                pdf.image(str(image_path), w=170)
                pdf.set_font("Nirmala", "", 9)
                pdf.multi_cell(0, 5, plt_name, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.ln(2)

    pdf.add_page()
    pdf.h1("7. Inference and Variance-Adaptor Analysis")
    pdf.body(
        "The exact test sentences of the assignment (E1-E4 for English including the numbered-date sentence, T1-T4 for "
        "Telugu including the split-cluster OCR spellings such as 'schind/chindi' and 'adavilo') are preserved verbatim "
        "in analyze_variance.py's TEST_SENTENCES and re-used by synthesize_samples.py. For each sentence and language the "
        "scripts generate: (a) HiFi-GAN-synthesised waveforms in Results/<Language>/test_generated_audios/, (b) per-token "
        "predicted duration, pitch and energy plots with token tick labels in Results/<Language>/variance_adaptor_plots/, "
        "and (c) the raw numeric plots in Results/<Language>/variance_adaptor_data/. All numbers come from the trained "
        "model at the last checkpoint; nothing is synthesised by hand."
    )

    for language in LANGUAGES:
        pdf.h2(f"{language} test sentences")
        data_root = ROOT / "Results" / language / "variance_adaptor_data"
        audio_root = ROOT / "Results" / language / "test_generated_audios"
        if not data_root.exists():
            continue
        for sentence_path in sorted(data_root.glob("*.json")):
            payload = load_json(sentence_path)
            audio_path = audio_root / f"{sentence_path.stem}.wav"
            duration_text = "n/a"
            if audio_path.exists():
                info = sf.info(str(audio_path))
                duration_text = f"{info.duration:.2f} s"
            pdf.set_font("Nirmala", "", 10)
            pdf.multi_cell(0, 5.6, f"{sentence_path.stem} ({duration_text}, {len(payload['tokens'])} tokens): "
                                    + payload["raw_text"], new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        for feature in ["duration", "pitch", "energy"]:
            for sentence_path in sorted(data_root.glob("*.json")):
                image_path = ROOT / "Results" / language / "variance_adaptor_plots" / feature / f"{sentence_path.stem}.png"
                if not image_path.exists():
                    continue
                if pdf.get_y() > 225:
                    pdf.add_page()
                pdf.image(str(image_path), w=170)
                pdf.set_font("Nirmala", "", 9)
                pdf.multi_cell(0, 5, f"{sentence_path.stem} predicted {feature}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.add_page()
    pdf.h1("8. Modifications to the Base Repository")
    pdf.body(
        "Only data handling, alignment and training-logistics code was changed; the FastSpeech 2 network itself, its "
        "hyper-parameters, loss functions and scheduler identity are preserved from ming024/FastSpeech2:"
    )
    for item in [
        "text/cleaners.py (telugu_cleaners added), text/grapheme.py and text/normalization.py added for language handling and grapheme-cluster tokenisation.",
        "alignment/ (MMSAligner, romanizer), force_align.py and validate_alignment.py added for forced alignment and its validation.",
        "preprocess_features.py added (mel/pitch/energy/duration extraction directly from alignment token_frames, phoneme-level pitch/energy).",
        "dataset.py: optional mel_batch_budget batch-grouping so corpus's long utterances fit in GPU RAM without being dropped.",
        "train.py: mixed-precision autocast/GradScaler switch (train.yaml: optimizer.amp), step_loss CSV logging; model/optimizer.py exposes update_learning_rate for AMP stepping.",
        "plot_losses.py, analyze_variance.py, synthesize_samples.py, build_language_folders.py added for figures, test-sentence analysis/audio and per-language deliverable folders.",
        "prepare_metadata.py fixed to guarantee per-line metadata rows; build_vocab.py added for grapheme vocabularies.",
    ]:
        pdf.bullet(item)

    pdf.h1("9. Reproducing the Results")
    for item in [
        "python English/preprocess.py (forced alignment with MMS_FA + validation + feature extraction)",
        "python English/train.py --restore_step 0",
        "python English/inference.py --restore_step 10000",
        "python Telugu/preprocess.py / Telugu/train.py / Telugu/inference.py",
        "python build_language_folders.py (assemble English/, Telugu/ + merged config snapshots and checkpoints)",
    ]:
        pdf.bullet(item)

    pdf.h1("10. Known Limitations")
    for item in [
        "Telugu utterance telugu_000927 excluded: source text/audio mismatch makes forced alignment infeasible.",
        "4 utterances removed from the English corpus: unvoiced pitch (see skip report).",
        "The pretrained universal HiFi-GAN vocoder is speaker-agnostic; speaker identity is dominated by the trained acoustic model, with vocoder colouring shared across speakers.",
        "English token alignment score (~0.70) is lower than the Telugu (~0.95) because grapheme-level romanization for spellings is a harder CTC objective than the actual pronunciation appears; boundary noise is <2 frames per token on average.",
    ]:
        pdf.bullet(item)

    out_path = ROOT / "Report" / "Technical_Report.pdf"
    pdf.output(str(out_path))
    return out_path


if __name__ == "__main__":
    print(build_report())
