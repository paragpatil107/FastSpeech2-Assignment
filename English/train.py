import argparse
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
    args = parser.parse_args()

    subprocess.run(
        [
            str(PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"),
            str(PROJECT_ROOT / "train.py"),
            "--restore_step",
            str(args.restore_step),
            "-p",
            str(PROJECT_ROOT / "config" / "en" / "preprocess.yaml"),
            "-m",
            str(PROJECT_ROOT / "config" / "en" / "model.yaml"),
            "-t",
            str(PROJECT_ROOT / "config" / "en" / "train.yaml"),
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
