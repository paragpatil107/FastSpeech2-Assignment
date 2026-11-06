import argparse
import shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent

LANGUAGE_CONFIG_DIRS = {"English": ROOT / "config" / "en", "Telugu": ROOT / "config" / "te"}


def merge_configs(config_dir):
    preprocess_config = yaml.load(open(config_dir / "preprocess.yaml", encoding="utf-8"), Loader=yaml.FullLoader)
    model_config = yaml.load(open(config_dir / "model.yaml", encoding="utf-8"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(config_dir / "train.yaml", encoding="utf-8"), Loader=yaml.FullLoader)
    return {"preprocess": preprocess_config, "model": model_config, "train": train_config}


def write_merged_config(language, output_dir):
    config_dir = LANGUAGE_CONFIG_DIRS[language]
    merged = merge_configs(config_dir)
    with open(output_dir / "config.yaml", "w", encoding="utf-8") as file:
        yaml.dump(merged, file, allow_unicode=True, sort_keys=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--languages", nargs="+", choices=["English", "Telugu"], default=["English", "Telugu"]
    )
    args = parser.parse_args()

    for language in args.languages:
        output_dir = ROOT / language
        output_dir.mkdir(parents=True, exist_ok=True)
        write_merged_config(language, output_dir)

        (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        ckpt_subdir = {
            "English": ROOT / "output" / "ckpt" / "en",
            "Telugu": ROOT / "output" / "ckpt" / "te",
        }[language]
        if ckpt_subdir.exists():
            for checkpoint in sorted(ckpt_subdir.glob("*.pth.tar")):
                shutil.copy2(checkpoint, output_dir / "checkpoints" / checkpoint.name)
        print(f"{language}: folder prepared")


if __name__ == "__main__":
    main()
