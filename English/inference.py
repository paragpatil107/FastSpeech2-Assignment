import argparse
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, required=True)
    parser.add_argument("--skip_audio", action="store_true")
    parser.add_argument("--skip_variance", action="store_true")
    args = parser.parse_args()

    config_dir = PROJECT_ROOT / "config" / "en"
    common = ["--language", "English", "--config_dir", str(config_dir), "--restore_step", str(args.restore_step)]

    if not args.skip_audio:
        subprocess.run([str(VENV_PYTHON), str(PROJECT_ROOT / "synthesize_samples.py")] + common, cwd=PROJECT_ROOT, check=True)
    if not args.skip_variance:
        subprocess.run([str(VENV_PYTHON), str(PROJECT_ROOT / "analyze_variance.py")] + common, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
