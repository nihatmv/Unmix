"""
Step 1: load one audio clip and look at it.

Run from anywhere:
    .venv/bin/python src/visualize_original.py            # first clip in input/
    .venv/bin/python src/visualize_original.py my.wav     # a specific file

It prints basic facts about the clip and saves output/original.png with:
    1. the waveform    (time vs amplitude)
    2. the spectrogram (time vs frequency, loudness in dB)
"""

import matplotlib

# "Agg" draws straight to an image file instead of opening a window.
# It must be selected before pyplot is imported.
matplotlib.use("Agg")

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from common import HOP_LENGTH, N_FFT, OUTPUT_DIR, find_input_file


def main():
    audio_path = find_input_file()

    # -----------------------------------------------------------------------
    # 1. Load the audio
    # -----------------------------------------------------------------------
    # Digital audio is just a long list of numbers ("samples"). Each number is
    # the air pressure at one instant, measured many thousands of times per
    # second. How many times per second is the "sample rate".
    #
    #   sr=None   -> keep the file's own sample rate (e.g. 44100 Hz) instead of
    #                letting librosa resample it down to 22050 Hz.
    #   mono=True -> average left and right into a single channel, so we have
    #                one simple 1-D array to work with.
    #
    # y  is the array of samples, as floats between -1.0 and +1.0.
    # sr is the sample rate in Hz.
    y, sr = librosa.load(audio_path, sr=None, mono=True)

    # -----------------------------------------------------------------------
    # 2. Print basic info
    # -----------------------------------------------------------------------
    # Duration follows directly from the two numbers above:
    # (how many samples we have) / (how many samples make up one second).
    num_samples = len(y)
    duration = num_samples / sr

    print(f"File:         {audio_path.name}")
    print(f"Sample rate:  {sr} Hz")
    print(f"Samples:      {num_samples}")
    print(f"Duration:     {duration:.2f} seconds")

    # -----------------------------------------------------------------------
    # 3. Compute the spectrogram
    # -----------------------------------------------------------------------
    # The waveform tells us HOW LOUD the sound is at each moment, but not WHICH
    # pitches are in it. To see pitches we use the Short-Time Fourier Transform:
    #
    #   - cut the audio into short overlapping slices (N_FFT samples each),
    #   - for each slice, run an FFT, which answers "how much of each frequency
    #     is present in this slice?",
    #   - stack the answers side by side.
    #
    # The result is a 2-D grid: rows are frequencies, columns are moments in
    # time. Voice and music overlap completely in the waveform, but in this grid
    # they often land in different cells, which is what makes separation possible.
    stft = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)

    # Each cell of the STFT is a complex number holding two things:
    #   magnitude -> how strong that frequency is  (what we want to see)
    #   phase     -> where in its cycle the wave is (needed later to turn the
    #                grid back into sound, but not useful to look at)
    # np.abs() keeps just the magnitude.
    magnitude = np.abs(stft)

    # Convert to decibels (dB). Our ears hear loudness on a ratio scale: a sound
    # with 10x the amplitude doesn't sound 10x louder. Raw magnitudes would show
    # only the few loudest peaks and leave everything else black. dB compresses
    # the range so quiet detail becomes visible.
    #
    # ref=np.max makes the loudest cell 0 dB, so every other value is negative:
    # "this many dB quieter than the loudest point in the clip".
    magnitude_db = librosa.amplitude_to_db(magnitude, ref=np.max)

    # -----------------------------------------------------------------------
    # 4. Plot waveform and spectrogram, stacked
    # -----------------------------------------------------------------------
    # Two rows for the plots, plus a narrow second column that holds the
    # colorbar. Giving the colorbar its own column keeps both plots the same
    # width, so a moment in time sits at the same x position in each.
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 8),
        width_ratios=[40, 1],
        layout="constrained",
    )
    ax_wave, ax_spec, ax_colorbar = axes[0, 0], axes[1, 0], axes[1, 1]
    axes[0, 1].axis("off")  # the waveform has no colour scale, leave this empty

    # Top: the waveform. x = time in seconds, y = amplitude of each sample.
    librosa.display.waveshow(y, sr=sr, ax=ax_wave)
    ax_wave.set_title(f"Waveform: {audio_path.name}")
    ax_wave.set_xlabel("Time (s)")
    ax_wave.set_ylabel("Amplitude")
    ax_wave.set_xlim(0, duration)

    # Bottom: the spectrogram. x = time, y = frequency, colour = loudness in dB.
    # y_axis="log" spaces the frequency axis by octaves, the way we hear pitch.
    # On a plain linear axis, speech and bass (mostly below ~4000 Hz) would be
    # squashed into a thin strip at the bottom.
    # hop_length must match the STFT above so the time axis comes out right.
    image = librosa.display.specshow(
        magnitude_db,
        sr=sr,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="log",
        ax=ax_spec,
    )
    ax_spec.set_title("Spectrogram")
    ax_spec.set_xlabel("Time (s)")
    ax_spec.set_ylabel("Frequency (Hz)")
    ax_spec.set_xlim(0, duration)
    fig.colorbar(image, cax=ax_colorbar, format="%+2.0f dB", label="Loudness")

    # -----------------------------------------------------------------------
    # 5. Save the figure
    # -----------------------------------------------------------------------
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / "original.png"
    fig.savefig(output_path, dpi=150)
    print(f"Saved plot:   {output_path}")


if __name__ == "__main__":
    main()
