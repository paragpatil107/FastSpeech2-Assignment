import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_loss(csv_path, column, title, ylabel, output_path):
    frame = pd.read_csv(csv_path)
    frame = frame[pd.to_numeric(frame["step"], errors="coerce").notna()]
    frame = frame.astype({column: "float64", "step": "float64"})
    steps = frame["step"].to_numpy()
    values = frame[column].to_numpy()
    plt.figure(figsize=(9, 5))
    plt.plot(steps, values, label=column, color="#1f77b4")
    if len(steps) > 1:
        window = max(1, len(values) // 20)
        smoothed = pd.Series(values).rolling(window, min_periods=1).mean()
        plt.plot(steps, smoothed, label="moving average", color="#ff7f0e")
    plt.xlabel("Training step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", choices=["English", "Telugu"], required=True)
    parser.add_argument("--log_path", required=True)
    parser.add_argument("--results_root", default="Results")
    args = parser.parse_args()

    csv_path = Path(args.log_path) / "step_losses.csv"
    results_root = Path(args.results_root) / args.language
    results_root.mkdir(parents=True, exist_ok=True)

    columns = {
        "total_loss": "Training total loss",
        "mel_loss": "Training mel loss (L1)",
        "mel_postnet_loss": "Training mel PostNet loss (L1)",
        "pitch_loss": "Training pitch loss (MSE)",
        "energy_loss": "Training energy loss (MSE)",
        "duration_loss": "Training duration loss (MSE)",
    }
    column_output_names = {
        "total_loss": "training_loss",
        "mel_loss": "training_mel_loss",
        "mel_postnet_loss": "training_mel_postnet_loss",
        "pitch_loss": "training_pitch_loss",
        "energy_loss": "training_energy_loss",
        "duration_loss": "training_duration_loss",
    }
    for column, title in columns.items():
        plot_loss(
            csv_path,
            column,
            f"{args.language} {title}",
            title.split("(")[-1].rstrip(")").strip("()").strip() or column,
            results_root / f"{column_output_names[column]}.png",
        )

    val_log_path = Path(args.log_path) / "val" / "log.txt"
    val_rows = []
    if val_log_path.exists():
        import re

        pattern = re.compile(
            r"Validation Step (\d+), Total Loss: ([0-9.]+), Mel Loss: ([0-9.]+), "
            r"Mel PostNet Loss: ([0-9.]+), Pitch Loss: ([0-9.]+), "
            r"Energy Loss: ([0-9.]+), Duration Loss: ([0-9.]+)"
        )
        for line in open(val_log_path, encoding="utf-8"):
            match = pattern.match(line.strip())
            if match:
                val_rows.append([int(match.group(1))] + [float(match.group(i)) for i in range(2, 8)])
        if val_rows:
            import numpy as np

            values = np.array(val_rows)
            for index, column in enumerate(
                [
                    "total_loss",
                    "mel_loss",
                    "mel_postnet_loss",
                    "pitch_loss",
                    "energy_loss",
                    "duration_loss",
                ],
                start=1,
            ):
                plt.figure(figsize=(9, 5))
                plt.plot(values[:, 0], values[:, index], marker="o", color="#d62728")
                plt.xlabel("Training step")
                plt.ylabel(f"Validation {column}")
                plt.title(f"{args.language} validation {column}")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(results_root / f"validation_{column}.png")
                plt.close()
                if index == 1:
                    (results_root / "validation_loss.png").unlink(missing_ok=True)
                    (results_root / "validation_loss.png").write_bytes(
                        (results_root / "validation_total_loss.png").read_bytes()
                    )
            np.save(results_root / "validation_loss_history.npy", values)
            print(f"validation points: {len(val_rows)}")


if __name__ == "__main__":
    main()
