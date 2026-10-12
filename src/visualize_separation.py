"""
Step 3: look at what the separation did.

Run from anywhere:
    .venv/bin/python src/visualize_separation.py            # first clip in input/
    .venv/bin/python src/visualize_separation.py my.wav     # a specific file

It runs the separation again (so the picture always matches the current
settings in separate.py) and saves output/separation.png with seven panels
that all share one time axis:
    1-3. waveforms:    original, foreground, background
    4-6. spectrograms: original, foreground, background
    7.   the foreground mask
"""

import sys

import matplotlib

# "Agg" draws straight to an image file instead of opening a window.
# It must be selected before pyplot is imported.
matplotlib.use("Agg")

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from common import HOP_LENGTH, N_FFT, OUTPUT_DIR, find_input_file
from separate import separate_with_mask

# One colour per layer, used for its waveform.
COLOR_ORIGINAL = "#2a78d6"  # blue
COLOR_FOREGROUND = "#eb6834"  # orange
COLOR_BACKGROUND = "#1baf7a"  # green

# Colour scale for the mask: near-white at 0, deepening to dark orange at 1.
# It uses the foreground's colour, so "more orange" reads as "more foreground".
MASK_COLORMAP = LinearSegmentedColormap.from_list(
    "foreground_mask", ["#fcfcfb", "#f6b89c", COLOR_FOREGROUND, "#7a2a0c"]
)

# The quietest level the spectrograms show. Anything more than 80 dB below the
# loudest point of the original is drawn as black.
DB_FLOOR = -80


def to_db(audio, reference):
    """STFT magnitude of `audio` in dB, measured against a shared reference level."""
    magnitude = np.abs(librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH))
    return librosa.amplitude_to_db(magnitude, ref=reference, top_db=None)


def main():
    audio_path = find_input_file()
    audio, sr = librosa.load(audio_path, sr=None, mono=True)
    duration = len(audio) / sr

    print(f"File:         {audio_path.name}")
    print("Separating... (this can take a little while on long clips)")
    try:
        foreground, background, mask_foreground = separate_with_mask(audio, sr)
    except ValueError as error:
        sys.exit(str(error))

    layers = [
        ("Original", audio, COLOR_ORIGINAL),
        ("Foreground (non-repeating, likely speech)", foreground, COLOR_FOREGROUND),
        ("Background (repeating, likely music)", background, COLOR_BACKGROUND),
    ]

    # -----------------------------------------------------------------------
    # Figure layout
    # -----------------------------------------------------------------------
    # Seven rows of plots, plus a narrow second column for the colorbars so
    # every plot ends up the same width. sharex="col" links the time axes: one
    # second sits at the same x position in every panel, so you can read
    # straight down from a waveform to its spectrogram to the mask.
    print("Drawing...")
    plt.rcParams["font.size"] = 13  # the figure is big, so make the text big too
    fig, axes = plt.subplots(
        7,
        2,
        figsize=(16, 30),
        width_ratios=[50, 1],
        height_ratios=[1, 1, 1, 1.6, 1.6, 1.6, 1.6],
        sharex="col",
        layout="constrained",
    )
    plot_axes = axes[:, 0]
    colorbar_axes = axes[:, 1]
    for ax in colorbar_axes:
        ax.axis("off")  # switched back on below for the rows that need one

    fig.suptitle(f"Separation of {audio_path.name}", fontsize=20)

    # -----------------------------------------------------------------------
    # Rows 1-3: waveforms
    # -----------------------------------------------------------------------
    # All three use the same y-axis range, taken from the loudest sample in any
    # of them. That way a layer that looks half as tall really is half as loud;
    # if each panel zoomed to fit its own signal you couldn't compare them.
    peak = max(np.abs(signal).max() for _, signal, _ in layers)
    y_limit = peak * 1.05

    for ax, (name, signal, color) in zip(plot_axes[0:3], layers):
        librosa.display.waveshow(signal, sr=sr, ax=ax, color=color)
        ax.set_title(f"Waveform: {name}", fontsize=15, loc="left")
        ax.set_ylabel("Amplitude")
        ax.set_ylim(-y_limit, y_limit)
        ax.set_xlabel("")
        ax.grid(True, color="#e1e0d9", linewidth=0.6)

    # -----------------------------------------------------------------------
    # Rows 4-6: spectrograms
    # -----------------------------------------------------------------------
    # Same idea as the shared y-axis above, but for colour: all three are
    # measured in dB against the loudest cell of the ORIGINAL, and drawn with
    # the same colour range. A given colour means the same loudness in every
    # panel, so a layer that looks darker really is quieter.
    reference = np.abs(librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH)).max()

    for ax, colorbar_ax, (name, signal, _) in zip(
        plot_axes[3:6], colorbar_axes[3:6], layers
    ):
        image = librosa.display.specshow(
            to_db(signal, reference),
            sr=sr,
            hop_length=HOP_LENGTH,
            x_axis="time",
            y_axis="log",
            vmin=DB_FLOOR,
            vmax=0,
            cmap="magma",
            ax=ax,
        )
        ax.set_title(f"Spectrogram: {name}", fontsize=15, loc="left")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_xlabel("")
        colorbar_ax.axis("on")
        fig.colorbar(
            image,
            cax=colorbar_ax,
            format="%+2.0f dB",
            label="Loudness (relative to original's peak)",
        )

    # -----------------------------------------------------------------------
    # Row 7: the foreground mask
    # -----------------------------------------------------------------------
    # The mask has one number per spectrogram cell, so it is drawn on the same
    # time and frequency axes as the spectrograms above. Each cell shows what
    # fraction of the original's energy at that time and frequency was let
    # through into the foreground layer:
    #   near 1 (dark orange) -> kept in the foreground
    #   near 0 (white)       -> removed from it, i.e. treated as background
    #
    # Compare it with the original spectrogram: the dark patches of the mask are
    # the parts of the picture the method decided did not repeat.
    ax_mask = plot_axes[6]
    image = librosa.display.specshow(
        mask_foreground,
        sr=sr,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="log",
        vmin=0,
        vmax=1,
        cmap=MASK_COLORMAP,
        ax=ax_mask,
    )
    ax_mask.set_title(
        "Foreground mask: how much of each cell went to the foreground",
        fontsize=15,
        loc="left",
    )
    ax_mask.set_ylabel("Frequency (Hz)")
    ax_mask.set_xlabel("Time (s)")
    ax_mask.set_xlim(0, duration)
    colorbar_axes[6].axis("on")
    fig.colorbar(image, cax=colorbar_axes[6], label="0 = background    1 = foreground")

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / "separation.png"
    fig.savefig(output_path, dpi=120)
    print(f"Saved plot:   {output_path}")


if __name__ == "__main__":
    main()
