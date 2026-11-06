import re
import unicodedata

from text import cleaners as text_cleaners


_whitespace_re = re.compile(r"\s+")


def collapse_whitespace(text):
    return re.sub(_whitespace_re, " ", text).strip()


def normalize_text(text, language, cleaner_names=None):
    text = unicodedata.normalize("NFC", text)
    if language == "English":
        for name in cleaner_names or ["english_cleaners"]:
            text = getattr(text_cleaners, name)(text)
        return collapse_whitespace(text)
    return collapse_whitespace(text)
