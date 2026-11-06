import csv
import random
from pathlib import Path

from datasets import load_dataset, Audio


BASE = Path("raw_data/English")
AUDIO_DIR = BASE / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

TARGET_DURATION = 6 * 60 * 60
VALIDATION_DURATION = 60 * 60
TOTAL_DURATION = 0
records = []

dataset = load_dataset(
    "ai4bharat/Svarah",
    split="test",
    streaming=True
)

dataset = dataset.cast_column(
    "audio_filepath",
    Audio(decode=False)
)

for sample in dataset:
    duration = float(sample["duration"])

    if duration < 1:
        continue

    if TOTAL_DURATION + duration > TARGET_DURATION:
        break

    audio_bytes = sample["audio_filepath"]["bytes"]

    if not audio_bytes:
        continue

    filename = f"english_{len(records):06d}.wav"
    (AUDIO_DIR / filename).write_bytes(audio_bytes)

    records.append({
        "filename": filename,
        "text": sample["text"],
        "duration": duration
    })

    TOTAL_DURATION += duration

    if len(records) % 100 == 0:
        print(
            f"Extracted: {len(records)} | "
            f"Hours: {TOTAL_DURATION / 3600:.2f}"
        )

random.Random(42).shuffle(records)

train_records = []
val_records = []
val_duration = 0

for record in records:
    if val_duration < VALIDATION_DURATION:
        val_records.append(record)
        val_duration += record["duration"]
    else:
        train_records.append(record)


def save_metadata(filename, rows):
    with open(BASE / filename, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["filename", "text", "duration"])

        for record in rows:
            writer.writerow([
                record["filename"],
                record["text"],
                record["duration"]
            ])


save_metadata("train.csv", train_records)
save_metadata("val.csv", val_records)

print("\nExtraction completed")
print("Total files:", len(records))
print("Training files:", len(train_records))
print("Validation files:", len(val_records))
print("Training hours:", round(sum(x["duration"] for x in train_records) / 3600, 3))
print("Validation hours:", round(sum(x["duration"] for x in val_records) / 3600, 3))
print("Total hours:", round(TOTAL_DURATION / 3600, 3))