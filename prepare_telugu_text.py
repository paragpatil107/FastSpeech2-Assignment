import csv
from pathlib import Path

base = Path("raw_data/Telugu")
output = Path("text_corpus/te.txt")

with open(base / "train.csv", encoding="utf-8") as file:
    records = list(csv.DictReader(file))

texts = [
    record["text"].strip()
    for record in records
    if record["text"].strip()
]

output.write_text("\n".join(texts), encoding="utf-8")

print("Training transcripts:", len(texts))
print("Saved:", output)