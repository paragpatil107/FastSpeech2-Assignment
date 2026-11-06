import argparse
import csv
import json
import statistics
from pathlib import Path

import numpy as np


def summarise_language(language, preprocessed_root, vocab_path):
    report_path = preprocessed_root / language / "alignment" / "alignment_report.csv"
    rows = list(csv.DictReader(open(report_path, encoding="utf-8")))
    latest_rows = {}
    for row in rows:
        latest_rows[row["basename"]] = row
    rows = list(latest_rows.values())
    ok_rows = [row for row in rows if row["status"] == "ok"]
    error_rows = [row for row in rows if row["status"] != "ok"]

    coverage = [float(row["coverage"]) for row in ok_rows]
    scores = [float(row["mean_score"]) for row in ok_rows if row["mean_score"]]

    zero_token_fraction_list = []
    duration_sum_mismatch = []
    token_counts = []
    for row in ok_rows:
        split = row["split"]
        payload_path = preprocessed_root / language / "alignment" / split / f"{row['basename']}.json"
        alignment = json.load(open(payload_path, encoding="utf-8"))
        token_frames = alignment["token_frames"]
        durations = np.diff(np.array(token_frames))
        total_tokens = len(durations)
        zero_tokens = int(np.sum(durations == 0))
        zero_token_fraction_list.append(zero_tokens / total_tokens)
        token_counts.append(total_tokens)
        audio_seconds = alignment["num_samples"] / alignment["sample_rate"]
        token_seconds = (token_frames[-1] - token_frames[0]) * alignment["frame_duration"]
        duration_sum_mismatch.append((token_frames[-1] - token_frames[0]) * alignment["frame_duration"] / max(audio_seconds, 1e-9))

    vocab = json.load(open(vocab_path, encoding="utf-8"))
    from text.grapheme import split_graphemes

    unknown_token_counts = 0
    total_token_counts = 0
    vocabulary_input = Path("processed_data") / language
    from text.normalization import normalize_text

    for split in ["train", "val"]:
        with open(vocabulary_input / f"{split}.txt", encoding="utf-8") as file:
            for line in file:
                parts = line.rstrip("\n").split("|", 3)
                normalized = normalize_text(parts[3], language)
                for token in split_graphemes(normalized):
                    total_token_counts += 1
                    if token not in vocab:
                        unknown_token_counts += 1

    summary = {
        "language": language,
        "total_files": len(rows),
        "aligned_files": len(ok_rows),
        "error_files": len(error_rows),
        "error_samples": [row["basename"] + ": " + row["error"][:120] for row in error_rows[:10]],
        "mean_word_coverage": round(statistics.mean(coverage), 4) if coverage else None,
        "min_word_coverage": round(min(coverage), 4) if coverage else None,
        "mean_token_score": round(statistics.mean(scores), 4) if scores else None,
        "score_min": round(min(scores), 4) if scores else None,
        "score_max": round(max(scores), 4) if scores else None,
        "mean_tokens_per_utterance": round(statistics.mean(token_counts), 2) if token_counts else None,
        "mean_zero_duration_token_fraction": round(statistics.mean(zero_token_fraction_list), 4) if zero_token_fraction_list else None,
        "mean_alignment_duration_ratio": round(statistics.mean(duration_sum_mismatch), 4) if duration_sum_mismatch else None,
        "unknown_vocabulary_tokens": unknown_token_counts,
        "total_text_tokens": total_token_counts,
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", choices=["English", "Telugu"], required=True)
    args = parser.parse_args()

    preprocessed_root = Path("preprocessed_data")
    vocab_path = Path("config") / ("en" if args.language == "English" else "te") / "vocab.json"
    summary = summarise_language(args.language, preprocessed_root, vocab_path)
    output_path = preprocessed_root / args.language / "alignment_summary.json"
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
