import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_align", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if not args.skip_align:
        command = [str(VENV_PYTHON), str(PROJECT_ROOT / "force_align.py"), "--language", "Telugu"]
        if args.device:
            command += ["--device", args.device]
        if args.limit:
            command += ["--limit", str(args.limit)]
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)

    subprocess.run(
        [str(VENV_PYTHON), str(PROJECT_ROOT / "validate_alignment.py"), "--language", "Telugu"],
        cwd=PROJECT_ROOT,
        check=True,
    )
    subprocess.run(
        [str(VENV_PYTHON), str(PROJECT_ROOT / "preprocess_features.py"), str(PROJECT_ROOT / "config" / "te" / "preprocess.yaml"), "--language", "Telugu"],
        cwd=PROJECT_ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
