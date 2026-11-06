import numpy as np
import soundfile as sf
import torch
import torchaudio

from text.grapheme import split_graphemes
from alignment.romanizer import Romanizer


class MMSAligner:
    def __init__(self, device, sample_rate=16000):
        bundle = torchaudio.pipelines.MMS_FA
        self.model = bundle.get_model().to(device).eval()
        self.aligner = bundle.get_aligner()
        self.labels = bundle.get_labels()
        self.alignable_chars = {
            label for label in self.labels if label not in ("-", "*")
        }
        self.romanizer = Romanizer()
        self.device = device
        self.sample_rate = sample_rate

    def load_waveform(self, wav_path):
        audio, sample_rate = sf.read(str(wav_path), dtype="float32", always_2d=True)
        waveform = torch.from_numpy(audio.T)
        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sample_rate, new_freq=self.sample_rate
            )
        return waveform, self.sample_rate

    def compute_emission(self, waveform, max_chunk_samples=576000):
        length = waveform.shape[1]
        if length <= max_chunk_samples:
            with torch.no_grad():
                outputs = self.model(waveform.to(self.device))
            if isinstance(outputs, tuple):
                outputs, _ = outputs
            return outputs[0].detach().cpu()

        emissions = []
        for start in range(0, length, max_chunk_samples):
            chunk = waveform[:, start : start + max_chunk_samples]
            with torch.no_grad():
                outputs = self.model(chunk.to(self.device))
            if isinstance(outputs, tuple):
                outputs, _ = outputs
            emissions.append(outputs[0].detach().cpu())
        return torch.cat(emissions, dim=0)

    def romanize_word(self, word):
        graphemes = split_graphemes(word)
        if not graphemes:
            return graphemes, "", None, "empty"

        cluster_roman = self.romanizer.romanize_graphemes(graphemes)
        word_roman = self.romanizer.romanize(word)

        if "".join(cluster_roman) == word_roman and all(
            roman for roman in cluster_roman
        ):
            return graphemes, word_roman, cluster_roman, "cluster"

        prefix_roman = []
        position = 0
        for roman in cluster_roman:
            if roman and word_roman.startswith(roman, position):
                prefix_roman.append(roman)
                position += len(roman)
            else:
                prefix_roman.append(None)
        if all(roman is not None for roman in prefix_roman):
            return graphemes, word_roman, prefix_roman, "prefix"

        return graphemes, word_roman, None, "word"

    def prepare_words(self, text):
        prepared_words = []
        for word in text.split():
            graphemes, word_roman, cluster_roman, mode = self.romanize_word(word)
            cluster_chars = []
            if cluster_roman is not None:
                for cluster_index, roman in enumerate(cluster_roman):
                    for char in roman:
                        if char in self.alignable_chars:
                            cluster_chars.append((char, cluster_index))
            token_ids = [self.labels.index(char) for char, _ in cluster_chars]
            if cluster_roman is None:
                token_ids = [
                    self.labels.index(char)
                    for char in word_roman
                    if char in self.alignable_chars
                ]
            prepared_words.append(
                {
                    "word": word,
                    "graphemes": list(graphemes),
                    "roman": word_roman,
                    "mode": mode,
                    "cluster_chars": cluster_chars,
                    "token_ids": token_ids,
                    "token_spans": [],
                    "start": None,
                    "end": None,
                    "score": None,
                }
            )
        return prepared_words

    def align(self, wav_path, text):
        waveform, sample_rate = self.load_waveform(wav_path)
        emission = self.compute_emission(waveform)
        num_frames = int(emission.shape[0])
        frame_time = waveform.shape[1] / (num_frames * sample_rate)

        prepared_words = self.prepare_words(text)

        token_lists = [word["token_ids"] for word in prepared_words if word["token_ids"]]
        aligned_spans = []
        if token_lists:
            with torch.no_grad():
                aligned_spans = self.aligner(emission, token_lists)

        span_position = 0
        for word in prepared_words:
            if word["token_ids"]:
                word["token_spans"] = [
                    span for span in aligned_spans[span_position] if span.end != -1
                ]
                span_position += 1
            if word["token_spans"]:
                word["start"] = int(min(span.start for span in word["token_spans"]))
                word["end"] = int(max(span.end for span in word["token_spans"]))
                word["score"] = float(
                    np.mean([span.score for span in word["token_spans"]])
                )

        self.interpolate_unaligned_words(prepared_words)
        boundaries = self.assign_token_frames(prepared_words, num_frames)

        words_report = []
        for word in prepared_words:
            words_report.append(
                {
                    "word": word["word"],
                    "graphemes": word["graphemes"],
                    "roman": word["roman"],
                    "mode": word["mode"],
                    "start_frame": word["start"],
                    "end_frame": word["end"],
                    "score": word["score"],
                    "token_count": len(word["token_spans"]),
                }
            )

        aligned_words = sum(1 for word in prepared_words if word["score"] is not None)
        word_scores = [word["score"] for word in prepared_words if word["score"] is not None]
        return {
            "sample_rate": sample_rate,
            "num_samples": int(waveform.shape[1]),
            "emission_frames": num_frames,
            "frame_duration": float(frame_time),
            "text": text,
            "words": words_report,
            "utterance_start_frame": int(boundaries[0]),
            "utterance_end_frame": int(boundaries[-1]),
            "token_frames": boundaries,
            "coverage": float(aligned_words / max(len(prepared_words), 1)),
            "mean_score": float(np.mean(word_scores)) if word_scores else None,
        }

    def interpolate_unaligned_words(self, prepared_words):
        aligned_positions = [
            index
            for index, word in enumerate(prepared_words)
            if word["start"] is not None
        ]
        for position, word in enumerate(prepared_words):
            if word["start"] is not None:
                continue
            lower = [index for index in aligned_positions if index < position]
            upper = [index for index in aligned_positions if index > position]
            if lower and upper:
                previous_end = prepared_words[lower[-1]]["end"]
                next_start = prepared_words[upper[0]]["start"]
                group = prepared_words[lower[-1] + 1 : upper[0]]
                group_counts = [max(len(item["graphemes"]), 1) for item in group]
                total = sum(group_counts)
                consumed = sum(
                    group_counts[: position - lower[-1] - 1]
                )
                start_frame = previous_end + round(
                    (next_start - previous_end) * consumed / total
                )
                end_frame = start_frame + round(
                    (next_start - previous_end) * max(len(word["graphemes"]), 1) / total
                )
                word["start"] = int(start_frame)
                word["end"] = int(end_frame)
            elif lower:
                word["start"] = word["end"] = int(prepared_words[lower[-1]]["end"])
            elif upper:
                word["start"] = word["end"] = int(prepared_words[upper[0]]["start"])

    def assign_token_frames(self, prepared_words, num_frames):
        text = " ".join(word["word"] for word in prepared_words)
        tokens = split_graphemes(text)
        num_tokens = len(tokens)
        boundaries = [None] * (num_tokens + 1)

        token_position = 0
        for word in prepared_words:
            graphemes = word["graphemes"]
            count = len(graphemes)
            if count == 0:
                continue
            if word["score"] is not None and word["mode"] in ("cluster", "prefix"):
                cluster_starts = {}
                for (char, cluster_index), span in zip(
                    word["cluster_chars"], word["token_spans"]
                ):
                    cluster_starts.setdefault(cluster_index, span.start)
                current = word["start"]
                for cluster_index in range(count):
                    proposed = cluster_starts.get(cluster_index, current)
                    boundaries[token_position + cluster_index] = proposed
                    current = max(current, proposed)
            elif word["score"] is not None:
                span_length = word["end"] - word["start"]
                for cluster_index in range(count):
                    boundaries[token_position + cluster_index] = word["start"] + round(
                        span_length * cluster_index / max(count, 1)
                    )
            else:
                span_length = word["end"] - word["start"]
                for cluster_index in range(count):
                    boundaries[token_position + cluster_index] = word["start"] + round(
                        span_length * cluster_index / max(count, 1)
                    )
            token_position += count
            while token_position < num_tokens and tokens[token_position] == " ":
                boundaries[token_position] = word["end"]
                token_position += 1

        resolved = []
        previous = 0
        for boundary in boundaries:
            current = boundary if boundary is not None else previous
            resolved.append(int(current))
            previous = current
        for index in range(1, len(resolved)):
            resolved[index] = max(resolved[index], resolved[index - 1])
        if resolved:
            final_aligned_end = max(
                [word["end"] for word in prepared_words if word["end"] is not None],
                default=0,
            )
            resolved[-1] = max(resolved[-1], int(final_aligned_end))
        return resolved
