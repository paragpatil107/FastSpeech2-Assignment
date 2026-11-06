import json
import unicodedata
from pathlib import Path

import regex


def split_graphemes(text):
    text = unicodedata.normalize("NFC", text)
    return regex.findall(r"\X", text)


def build_vocabulary(texts):
    vocabulary = {"<pad>": 0, "<unk>": 1}

    graphemes = sorted({
        grapheme
        for text in texts
        for grapheme in split_graphemes(text)
    })

    for grapheme in graphemes:
        if grapheme not in vocabulary:
            vocabulary[grapheme] = len(vocabulary)

    return vocabulary


def save_vocabulary(vocabulary, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(vocabulary, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def load_vocabulary(path):
    return json.loads(
        Path(path).read_text(encoding="utf-8")
    )


def text_to_grapheme_ids(text, vocabulary):
    return [
        vocabulary.get(grapheme, vocabulary["<unk>"])
        for grapheme in split_graphemes(text)
    ]


def grapheme_ids_to_text(ids, vocabulary):
    reverse_vocabulary = {
        index: grapheme
        for grapheme, index in vocabulary.items()
    }

    return "".join(
        reverse_vocabulary.get(index, "")
        for index in ids
        if index not in (0, 1)
    )