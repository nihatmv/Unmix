"""
Compare the two separation methods side by side.

`separate.py --method both` draws this picture for you. To redraw it from the
WAV files that run left in output/repet/ and output/demucs/, without
separating again:
    .venv/bin/python src/visualize_comparison.py

It saves output/comparison.png with five waveforms stacked on one time axis:
    1.   original
    2-3. REPET foreground and background
    4-5. Demucs foreground and background
"""

import sys

import matplotlib

# "Agg" draws straight to an image file instead of opening a window.
# It must be selected before pyplot is imported.
matplotlib.use("Agg")

import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from common import OUTPUT_DIR

# One colour per layer, the same as in the other plots and the viewer.
COLOR_ORIGINAL = "#2a78d6"  # blue
COLOR_FOREGROUND = "#eb6834"  # orange
COLOR_BACKGROUND = "#1baf7a"  # green


def save_comparison(audio, repet_layers, demucs_layers, sr, clip_name):
    """Draw the five waveforms and save output/comparison.png.

    audio:         the original mono signal.
    repet_layers:  (foreground, background) from the REPET method.
    demucs_layers: (foreground, background) from Demucs.
    """
    rows = [
        ("Original", audio, COLOR_ORIGINAL),
        ("REPET foreground", repet_layers[0], COLOR_FOREGROUND),
        ("REPET background", repet_layers[1], COLOR_BACKGROUND),
        ("Demucs foreground", demucs_layers[0], COLOR_FOREGROUND),
        ("Demucs background", demucs_layers[1], COLOR_BACKGROUND),
    ]

    # All five panels use the same y-axis range, taken from the loudest sample
    # in any of them. That way a layer that looks half as tall really is half
    # as loud, and the two methods can be compared by eye.
    peak = max(np.abs(signal).max() for _, signal, _ in rows)
    y_limit = peak * 1.05

    # sharex links the time axes, so one second sits at the same x position in
    # every panel and you can read straight down the page.
    plt.rcParams["font.size"] = 13
    fig, axes = plt.subplots(5, 1, figsize=(16, 15), sharex=True, layout="constrained")
    fig.suptitle(f"REPET and Demucs on {clip_name}", fontsize=20)

    for ax, (name, signal, color) in zip(axes, rows):
        librosa.display.waveshow(signal, sr=sr, ax=ax, color=color)
        ax.set_title(f"Waveform: {name}", fontsize=15, loc="left")
        ax.set_ylabel("Amplitude")
        ax.set_ylim(-y_limit, y_limit)
        ax.set_xlabel("")
        ax.grid(True, color="#e1e0d9", linewidth=0.6)

    axes[-1].set_xlabel("Time (s)")
    axes[-1].set_xlim(0, len(audio) / sr)

    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / "comparison.png"
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"Saved plot:   {output_path}")


def main():
    repet_dir = OUTPUT_DIR / "repet"
    demucs_dir = OUTPUT_DIR / "demucs"
    if not (repet_dir / "foreground.wav").is_file() or not (
        demucs_dir / "foreground.wav"
    ).is_file():
        sys.exit("Nothing to compare yet. Run: src/separate.py --method both")

    # A run with a single method only refreshes its own folder, so the two can
    # hold different clips. Comparing those would be meaningless.
    clip_name = (repet_dir / "source.txt").read_text().strip()
    demucs_clip_name = (demucs_dir / "source.txt").read_text().strip()
    if clip_name != demucs_clip_name:
        sys.exit(
            f"output/repet/ holds {clip_name} but output/demucs/ holds "
            f"{demucs_clip_name}. Run: src/separate.py --method both"
        )

    # Both folders hold the same original, so read it from either one.
    audio, sr = sf.read(repet_dir / "original.wav")
    repet_layers = [sf.read(repet_dir / f"{name}.wav")[0] for name in ("foreground", "background")]
    demucs_layers = [sf.read(demucs_dir / f"{name}.wav")[0] for name in ("foreground", "background")]

    save_comparison(audio, repet_layers, demucs_layers, sr, clip_name)


if __name__ == "__main__":
    main()
