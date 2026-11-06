import argparse
from pathlib import Path

from text.grapheme import build_vocabulary, save_vocabulary


parser = argparse.ArgumentParser()
parser.add_argument("--language", choices=["en", "te"], required=True)
parser.add_argument("--input", required=True)
args = parser.parse_args()

input_path = Path(args.input)
texts = [
    line.strip()
    for line in input_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

vocabulary = build_vocabulary(texts)

output_path = Path("config") / args.language / "vocab.json"
save_vocabulary(vocabulary, output_path)

print("Language:", args.language)
print("Vocabulary size:", len(vocabulary))
print("Saved:", output_path)