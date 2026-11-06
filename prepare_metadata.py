
import argparse
import csv
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--language", choices=["English", "Telugu"], required=True)
args = parser.parse_args()

base = Path("raw_data") / args.language
output_dir = Path("processed_data") / args.language
output_dir.mkdir(parents=True, exist_ok=True)

for split in ["train", "val"]:
    with open(base / f"{split}.csv", encoding="utf-8") as file:
        records = list(csv.DictReader(file))

    output_path = output_dir / f"{split}.txt"

    with open(output_path, "w", encoding="utf-8") as file:
        for record in records:
            filename = Path(record["filename"]).stem
            text = record["text"].strip()
            text = text.replace("\r", " ").replace("\n", " ")
            file.write(f"{filename}|speaker_0|{text}|{text}\n")

    print(f"{args.language} {split}: {len(records)} records")
    print(f"Saved: {output_path}")
