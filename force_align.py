import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch

from alignment.mms_align import MMSAligner
from text.normalization import normalize_text

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_meta(path, split):
    with open(path, encoding="utf-8") as file:
        lines = [line.rstrip("\n") for line in file if line.strip()]
    entries = []
    for line in lines:
        basename, speaker, text, raw_text = line.split("|", 3)
        entries.append((basename, text, split))
    return entries


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", choices=["English", "Telugu"], required=True)
    parser.add_argument("--out_root", default="preprocessed_data")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--split", default=None, choices=["train", "val"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    aligner = MMSAligner(args.device)

    meta_dir = Path("processed_data") / args.language
    entries = []
    splits = [args.split] if args.split else ["train", "val"]
    for split in splits:
        entries += load_meta(meta_dir / f"{split}.txt", split)

    out_dir = Path(args.out_root) / args.language / "alignment"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.limit is not None:
        entries = entries[: args.limit]

    report_path = out_dir / "alignment_report.csv"
    processed_entries = set()
    if report_path.exists():
        with open(report_path, encoding="utf-8") as file:
            reader = csv.DictReader(file)
            processed_entries = {
                row["basename"] for row in reader if row["status"] == "ok"
            }

    report_file = open(report_path, "a", encoding="utf-8", newline="")
    writer = csv.writer(report_file)
    if report_path.exists() and report_path.stat().st_size == 0:
        writer.writerow(
            [
                "basename",
                "split",
                "word_count",
                "aligned_words",
                "coverage",
                "mean_score",
                "alignment_seconds",
                "audio_seconds",
                "status",
                "error",
            ]
        )

    for basename, text, split in entries:
        if basename in processed_entries:
            continue
        split_dir = out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        wav_path = Path("raw_data") / args.language / "audio" / f"{basename}.wav"
        start_time = time.time()
        try:
            normalized = normalize_text(text, args.language)
            report = aligner.align(str(wav_path), normalized)
            payload_path = split_dir / f"{basename}.json"
            with open(payload_path, "w", encoding="utf-8") as file:
                json.dump(report, file, ensure_ascii=False)
            writer.writerow(
                [
                    basename,
                    split,
                    len(report["words"]),
                    sum(
                        1
                        for word in report["words"]
                        if word["score"] is not None
                    ),
                    round(report["coverage"], 4),
                    round(report["mean_score"], 4) if report["mean_score"] is not None else "",
                    round(time.time() - start_time, 3),
                    round(report["num_samples"] / report["sample_rate"], 3),
                    "ok",
                    "",
                ]
            )
        except Exception as error:
            writer.writerow(
                [
                    basename,
                    split,
                    "",
                    "",
                    "",
                    "",
                    round(time.time() - start_time, 3),
                    "",
                    "error",
                    repr(error),
                ]
            )
        report_file.flush()

    report_file.close()


if __name__ == "__main__":
    main()
