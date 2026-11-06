import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.font_manager as font_manager

from dataset import load_text_settings
from text.grapheme import split_graphemes
from text.normalization import normalize_text
from utils.model import get_model

font_path = Path(__file__).resolve().parent / "Report" / "Nirmala_0.ttf"
if font_path.exists():
    font_manager.fontManager.addfont(str(font_path))
    plt.rcParams["font.family"] = ["DejaVu Sans", "Nirmala UI"]


TEST_SENTENCES = {
    "English": {
        "E1": "The antidisestablishmentarianism movement gained considerable momentum during the parliamentary proceedings.",
        "E2a": "The orchestra's conduct during the ceremony was praised by the audience.",
        "E2b": "The researchers will conduct an experiment to evaluate the proposed method.",
        "E3": "The rhyth  m of the orchestra changed unexpectedly.",
        "E4": "The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time.",
    },
    "Telugu": {
        "T1": "న్యూ రోఫార్మ కోలాజికల్  పరిశోధనలో నూతన ఔషధాల ప్రభావాన్ని  విశ్లేషించారు.",
        "T2a": "తాడుపై అతని పట్టు చాలా బలంగా ఉంది.",
        "T2b": "ఆమె పట్టు చీరను ధరించి వేడుకకు వచ్చ ింది .",
        "T3": "పులి అడ  విలో నిశ్శ బ్దంగా నడిచింది.",
        "T4": "ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది, సమయానికి సిద్ధంగా ఉంచండి.",
    },
}


def build_token_batch(text, language, preprocess_config):
    cleaned = normalize_text(
        text,
        language,
        preprocess_config["preprocessing"]["text"].get("text_cleaners"),
    )
    _, tokenization, vocabulary = load_text_settings(preprocess_config)
    if tokenization != "grapheme":
        raise ValueError("Only grapheme tokenization is supported for these languages.")
    token_labels = split_graphemes(cleaned)
    tokens = np.array(
        [vocabulary.get(grapheme, vocabulary["<unk>"]) for grapheme in token_labels]
    )
    return cleaned, tokens, token_labels


def denormalize(values, stats, key):
    _, _, mean, std = stats[key]
    return values * std + mean


def plot_feature(feature, values, token_labels, sentence_id, language, output_path):
    plt.figure(figsize=(max(10, len(values) * 0.5), 4.2))
    positions = np.arange(len(values))
    if feature == "duration":
        plt.bar(positions, values, color="#1f77b4")
        plt.ylabel("Predicted frames (86.13 frames/s)")
    elif feature == "pitch":
        plt.plot(positions, values, marker="o", markersize=4, color="#d62728")
        plt.ylabel("Predicted F0 (Hz)")
    else:
        plt.plot(positions, values, marker="s", markersize=4, color="#2ca02c")
        plt.ylabel("Predicted energy (linear scale)")
    short_labels = [
        label if len(label) <= 2 else label[0] + "e" for label in token_labels
    ]
    if len(short_labels) > 90:
        stride = max(1, len(positions) // 90)
        plt.xticks(
            positions[::stride], short_labels[::stride], rotation=90, fontsize="x-small"
        )
    else:
        plt.xticks(positions, short_labels, rotation=90, fontsize="x-small")
    plt.xlabel("Token (grapheme) sequence")
    plt.title(f"{language} {sentence_id}: predicted {feature} per token")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", choices=["English", "Telugu"], required=True)
    parser.add_argument("--restore_step", type=int, required=True)
    parser.add_argument("--config_dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--results_root", default="Results")
    args = parser.parse_args()

    language = args.language
    config_dir = Path(args.config_dir)
    preprocess_config = yaml.load(open(config_dir / "preprocess.yaml", encoding="utf-8"), Loader=yaml.FullLoader)
    model_config = yaml.load(open(config_dir / "model.yaml", encoding="utf-8"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(config_dir / "train.yaml", encoding="utf-8"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    class RestoreArgs:
        restore_step = args.restore_step

    model = get_model(RestoreArgs, configs, args.device, train=False)

    with open(Path(preprocess_config["path"]["preprocessed_path"]) / "stats.json", encoding="utf-8") as file:
        stats = json.load(file)

    language_root = Path(args.results_root) / language
    for feature in ["duration", "pitch", "energy"]:
        (language_root / "variance_adaptor_plots" / feature).mkdir(parents=True, exist_ok=True)
    (language_root / "variance_adaptor_data").mkdir(parents=True, exist_ok=True)

    for sentence_id, text in TEST_SENTENCES[language].items():
        cleaned, tokens, token_labels = build_token_batch(text, language, preprocess_config)
        text_tensor = torch.from_numpy(tokens).long().unsqueeze(0).to(args.device)
        src_lens = torch.tensor([len(tokens)], dtype=torch.int64, device=args.device)
        speakers = torch.zeros(1, dtype=torch.int64, device=args.device)
        with torch.no_grad():
            output = model(speakers, text_tensor, src_lens, int(src_lens.max()))

        lengths = int(src_lens.max())
        durations = output[5][0, :lengths].detach().cpu().numpy()
        pitch = output[2][0, :lengths].detach().cpu().numpy()
        energy = output[3][0, :lengths].detach().cpu().numpy()
        pitch_hz = denormalize(pitch, stats, "pitch")
        energy_raw = denormalize(energy, stats, "energy")

        payload_path = language_root / "variance_adaptor_data" / f"{sentence_id}.json"
        with open(payload_path, "w", encoding="utf-8") as file:
            json.dump(
                {
                    "sentence_id": sentence_id,
                    "raw_text": text,
                    "normalized_text": cleaned,
                    "tokens": token_labels,
                    "duration_frames": durations.tolist(),
                    "pitch_hz": pitch_hz.tolist(),
                    "energy_raw": energy_raw.tolist(),
                },
                file,
                ensure_ascii=False,
                indent=2,
            )

        for feature, values in [
            ("duration", durations),
            ("pitch", pitch_hz),
            ("energy", energy_raw),
        ]:
            plot_feature(
                feature,
                values,
                token_labels,
                sentence_id,
                language,
                language_root / "variance_adaptor_plots" / feature / f"{sentence_id}.png",
            )
        tqdm.write(
            f"{sentence_id}: tokens={len(tokens)}, tokens_mapped={(tokens != 1).sum()}/{len(tokens)}, "
            f"mean_duration={durations.mean():.2f}, mean_pitch={pitch_hz.mean():.1f} Hz"
        )


if __name__ == "__main__":
    main()
