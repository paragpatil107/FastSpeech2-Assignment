from text.grapheme import (
    split_graphemes,
    build_vocabulary,
    text_to_grapheme_ids,
    grapheme_ids_to_text
)

samples = {
    "en": "Hello, how are you?",
    "te": "నరకవాసి"
}

for language, text in samples.items():
    graphemes = split_graphemes(text)
    vocabulary = build_vocabulary([text])
    ids = text_to_grapheme_ids(text, vocabulary)
    reconstructed = grapheme_ids_to_text(ids, vocabulary)

    print(f"\nLanguage: {language}")
    print("Original:", text)
    print("Graphemes:", graphemes)
    print("Token count:", len(graphemes))
    print("Token IDs:", ids)
    print("Reconstructed:", reconstructed)
    print("Match:", text == reconstructed)