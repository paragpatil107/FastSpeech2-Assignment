import argparse
import json
import os
import sys
from pathlib import Path

import librosa
import numpy as np
import pyworld as pw
import torch
import yaml
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from text.grapheme import split_graphemes

import audio as Audio


def utterance_boundaries(alignment, sampling_rate):
    frame_duration = alignment["frame_duration"]
    start_sec = alignment["utterance_start_frame"] * frame_duration
    end_sec = min(
        (alignment["utterance_end_frame"] + 1) * frame_duration,
        alignment["num_samples"] / 16000.0,
    )
    local_seconds = (
        np.array(alignment["token_frames"], dtype=np.float64) * frame_duration
        - start_sec
    )
    local_seconds = np.maximum(local_seconds, 0.0)
    return local_seconds, start_sec, end_sec


def compute_features(wav, boundary_seconds, stft_module, config):
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    stft_config = config["preprocessing"]["stft"]

    mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav, stft_module)
    mel_frames = mel_spectrogram.shape[1]

    frames = np.round(boundary_seconds[:-1] * sampling_rate / stft_config["hop_length"])
    frames = np.maximum.accumulate(frames)
    frames = np.minimum(frames, mel_frames)
    token_frames = [int(frame) for frame in frames] + [mel_frames]
    durations = np.diff(np.array(token_frames, dtype=np.int64))

    pitch, times = pw.dio(
        wav.astype(np.float64),
        sampling_rate,
        frame_period=stft_config["hop_length"] / sampling_rate * 1000,
    )
    pitch = pw.stonemask(wav.astype(np.float64), pitch, times, sampling_rate)
    pitch = pitch[:mel_frames]
    if len(pitch) < mel_frames:
        pitch = np.pad(pitch, (0, mel_frames - len(pitch)), constant_values=0)

    energy = energy[:mel_frames]
    if len(energy) < mel_frames:
        energy = np.pad(energy, (0, mel_frames - len(energy)), constant_values=0)

    durations[-1] = mel_frames - token_frames[-2]
    if int(durations.sum()) != mel_frames:
        return None

    return durations, pitch, energy, mel_spectrogram.T


def phoneme_level_pitch(pitch, durations):
    nonzero_ids = np.where(pitch != 0)[0]
    if len(nonzero_ids) == 0:
        return np.zeros_like(pitch)
    if len(nonzero_ids) == 1:
        return np.full_like(pitch, pitch[nonzero_ids[0]], dtype=np.float64)
    interp_fn = interp1d(
        nonzero_ids,
        pitch[nonzero_ids],
        fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
        bounds_error=False,
    )
    pitch = interp_fn(np.arange(0, len(pitch)))

    averaged = np.zeros(len(durations))
    position = 0
    for index, duration in enumerate(durations):
        duration = int(duration)
        if duration > 0:
            averaged[index] = np.mean(pitch[position: position + duration])
        position += duration
    return averaged


def phoneme_level_energy(energy, durations):
    averaged = np.zeros(len(durations))
    position = 0
    for index, duration in enumerate(durations):
        duration = int(duration)
        if duration > 0:
            averaged[index] = np.mean(energy[position: position + duration])
        position += duration
    return averaged


def normalize_directory(directory, mean, std):
    max_value = np.finfo(np.float64).min
    min_value = np.finfo(np.float64).max
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        values = (np.load(filepath) - mean) / std
        np.save(filepath, values)
        if len(values):
            max_value = max(max_value, values.max())
            min_value = min(min_value, values.min())
    return min_value, max_value


def build_preprocessed_data(language, preprocess_config):
    preprocessed_path = Path(preprocess_config["path"]["preprocessed_path"])
    manifest_dir = Path("processed_data") / language
    alignment_root = preprocessed_path / "alignment"
    raw_path = Path(preprocess_config["path"]["raw_path"])

    for feature_dir in ["mel", "pitch", "energy", "duration"]:
        (preprocessed_path / feature_dir).mkdir(parents=True, exist_ok=True)

    stft_config = preprocess_config["preprocessing"]["stft"]
    audio_config = preprocess_config["preprocessing"]["audio"]
    stft_module = Audio.stft.TacotronSTFT(
        stft_config["filter_length"],
        stft_config["hop_length"],
        stft_config["win_length"],
        preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        audio_config["sampling_rate"],
        preprocess_config["preprocessing"]["mel"]["mel_fmin"],
        preprocess_config["preprocessing"]["mel"]["mel_fmax"],
    )

    meta_lines = {}
    for split in ["train", "val"]:
        with open(manifest_dir / f"{split}.txt", encoding="utf-8") as file:
            meta_lines[split] = [line.rstrip("\n") for line in file if line.strip()]

    pitch_samples = []
    energy_samples = []
    skipped = []
    processed = {split: [] for split in ["train", "val"]}
    total_frames = 0

    for split in ["train", "val"]:
        alignment_split_dir = alignment_root / split
        for line in tqdm(meta_lines[split], desc=f"{language} {split}"):
            basename, speaker, text, raw_text = line.split("|", 3)
            alignment_path = alignment_split_dir / f"{basename}.json"
            if not alignment_path.exists():
                skipped.append((basename, "missing_alignment"))
                continue
            alignment = json.load(open(alignment_path, encoding="utf-8"))
            token_text = alignment["text"]
            n_tokens = len(split_graphemes(token_text))
            if n_tokens != len(alignment["token_frames"]) - 1:
                skipped.append((basename, "token_mismatch"))
                continue

            boundary_seconds, start_sec, end_sec = utterance_boundaries(
                alignment, audio_config["sampling_rate"]
            )
            wav, _ = librosa.load(raw_path / f"{basename}.wav", sr=audio_config["sampling_rate"])
            wav = wav[int(start_sec * audio_config["sampling_rate"]): int(end_sec * audio_config["sampling_rate"])].astype(np.float32)
            if len(wav) < stft_config["filter_length"]:
                skipped.append((basename, "too_short"))
                continue

            features = compute_features(wav, boundary_seconds, stft_module, preprocess_config)
            if features is None:
                skipped.append((basename, "frame_mismatch"))
                continue
            durations, pitch, energy, mel = features
            if sum(pitch != 0) <= 1:
                skipped.append((basename, "unvoiced"))
                continue
            if preprocess_config["preprocessing"]["pitch"]["feature"] == "phoneme_level":
                pitch = phoneme_level_pitch(pitch, durations)
            if preprocess_config["preprocessing"]["energy"]["feature"] == "phoneme_level":
                energy = phoneme_level_energy(energy, durations)

            np.save(preprocessed_path / "duration" / f"{speaker}-duration-{basename}.npy", durations.astype(np.int64))
            np.save(preprocessed_path / "pitch" / f"{speaker}-pitch-{basename}.npy", pitch)
            np.save(preprocessed_path / "energy" / f"{speaker}-energy-{basename}.npy", energy.astype(np.float32))
            np.save(preprocessed_path / "mel" / f"{speaker}-mel-{basename}.npy", mel.astype(np.float32))

            voiced = pitch[pitch != 0]
            if len(voiced):
                pitch_samples.append(voiced)
            energy_samples.append(np.array(energy))
            total_frames += mel.shape[0]
            processed[split].append(f"{basename}|{speaker}|{token_text}|{raw_text}")

    print(f"Computing statistics for {language} ...")
    pitch_flat = np.concatenate(pitch_samples) if pitch_samples else np.zeros(1)
    energy_flat = np.concatenate(energy_samples) if energy_samples else np.zeros(1)
    scaler = StandardScaler()
    pitch_mean = float(scaler.fit(pitch_flat.reshape(-1, 1)).mean_[0])
    pitch_std = float(scaler.scale_[0])
    energy_mean = float(scaler.fit(energy_flat.reshape(-1, 1)).mean_[0])
    energy_std = float(scaler.scale_[0])

    if not preprocess_config["preprocessing"]["pitch"]["normalization"]:
        pitch_mean, pitch_std = 0.0, 1.0
    if not preprocess_config["preprocessing"]["energy"]["normalization"]:
        energy_mean, energy_std = 0.0, 1.0

    pitch_min, pitch_max = normalize_directory(preprocessed_path / "pitch", pitch_mean, pitch_std)
    energy_min, energy_max = normalize_directory(preprocessed_path / "energy", energy_mean, energy_std)

    hours = total_frames * stft_config["hop_length"] / audio_config["sampling_rate"] / 3600
    stats = {
        "pitch": [float(pitch_min), float(pitch_max), pitch_mean, pitch_std],
        "energy": [float(energy_min), float(energy_max), energy_mean, energy_std],
    }
    with open(preprocessed_path / "speakers.json", "w") as file:
        json.dump({"speaker_0": 0}, file)
    with open(preprocessed_path / "stats.json", "w") as file:
        json.dump(stats, file, indent=2)
    with open(preprocessed_path / "train.txt", "w", encoding="utf-8") as file:
        file.write("\n".join(processed["train"]) + "\n")
    with open(preprocessed_path / "val.txt", "w", encoding="utf-8") as file:
        file.write("\n".join(processed["val"]) + "\n")
    with open(preprocessed_path / "skip_report.json", "w") as file:
        json.dump(skipped, file, indent=2)
    with open(preprocessed_path / "corpus_stats.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                language: {
                    "train_utterances": len(processed["train"]),
                    "val_utterances": len(processed["val"]),
                    "mel_frames": int(total_frames),
                    "speech_hours": round(hours, 3),
                    "pitch_mean_hz": stats["pitch"][2] if pitch_std else 0.0,
                    "pitch_std_hz": stats["pitch"][3] if pitch_std else 1.0,
                    "energy_mean": stats["energy"][2],
                    "energy_std": stats["energy"][3],
                }
            },
            file,
            ensure_ascii=False,
            indent=2,
        )
    print(f"{language}: {len(processed['train'])} train, {len(processed['val'])} val, skipped {len(skipped)}, {hours:.2f} speech hours")
    return processed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--language", choices=["English", "Telugu"], required=True)
    args = parser.parse_args()
    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    build_preprocessed_data(args.language, config)


if __name__ == "__main__":
    main()
