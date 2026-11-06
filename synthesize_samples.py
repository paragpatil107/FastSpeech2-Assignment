import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_variance import TEST_SENTENCES, build_token_batch
from utils.model import get_vocoder, vocoder_infer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", choices=["English", "Telugu"], required=True)
    parser.add_argument("--restore_step", type=int, required=True)
    parser.add_argument("--config_dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--results_root", default="Results")
    parser.add_argument(
        "--pitch_control", type=float, default=1.0,
    )
    parser.add_argument(
        "--energy_control", type=float, default=1.0,
    )
    parser.add_argument(
        "--duration_control", type=float, default=1.0,
    )
    args = parser.parse_args()

    language = args.language
    config_dir = Path(args.config_dir)
    preprocess_config = yaml.load(open(config_dir / "preprocess.yaml", encoding="utf-8"), Loader=yaml.FullLoader)
    model_config = yaml.load(open(config_dir / "model.yaml", encoding="utf-8"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(config_dir / "train.yaml", encoding="utf-8"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    class RestoreArgs:
        restore_step = args.restore_step

    from utils.model import get_model

    model = get_model(RestoreArgs, configs, args.device, train=False)
    vocoder = get_vocoder(model_config, args.device)

    audio_root = Path(args.results_root) / language / "test_generated_audios"
    audio_root.mkdir(parents=True, exist_ok=True)
    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]

    for sentence_id, text in TEST_SENTENCES[language].items():
        cleaned, tokens, _ = build_token_batch(text, language, preprocess_config)
        text_tensor = torch.from_numpy(tokens).long().unsqueeze(0).to(args.device)
        src_lens = torch.tensor([len(tokens)], dtype=torch.int64, device=args.device)
        speakers = torch.zeros(1, dtype=torch.int64, device=args.device)
        with torch.no_grad():
            output = model(
                speakers,
                text_tensor,
                src_lens,
                int(src_lens.max()),
                p_control=args.pitch_control,
                e_control=args.energy_control,
                d_control=args.duration_control,
            )
        mel_predictions = output[1].transpose(1, 2)
        lengths = output[9] * preprocess_config["preprocessing"]["stft"]["hop_length"]
        wavs = vocoder_infer(
            mel_predictions, vocoder, model_config, preprocess_config, lengths=lengths
        )
        output_path = audio_root / f"{sentence_id}.wav"
        sf.write(str(output_path), wavs[0].astype(np.float32) / 32768.0, sampling_rate)
        print(f"{sentence_id}: {len(wavs[0]) / sampling_rate:.2f} s -> {output_path}")


if __name__ == "__main__":
    main()
