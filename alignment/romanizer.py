import unicodedata

from uroman import Uroman


class Romanizer:
    def __init__(self):
        self._uroman = Uroman()

    def romanize(self, text):
        text = unicodedata.normalize("NFC", text)
        return self._uroman.romanize_string(text)

    def romanize_graphemes(self, graphemes):
        return [self.romanize(grapheme) for grapheme in graphemes]
