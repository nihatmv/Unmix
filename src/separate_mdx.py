"""
A third way to split the clip: MDX-Net, a neural network trained for one job
only, pulling the human voice out of a mix.

You don't run this file directly. Use the --method flag of separate.py:
    .venv/bin/python src/separate.py --method mdx

It needs the extra package in requirements-mdx.txt and the model file
models/Kim_Vocal_2.onnx (see MODEL_FILE below).

How it differs from Demucs in separate_demucs.py: Demucs splits a song into
four stems (drums, bass, other, vocals) and works mostly on the raw waveform.
This model answers one question, "which part of this spectrogram is a voice?",
and it works on the spectrogram, like the REPET method does. Putting all its
effort into the voice is why it leaves less music behind in the voice layer.
"""

import sys

import librosa
import numpy as np
import onnxruntime

from common import PROJECT_DIR

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

# The trained model, as one file. "Kim_Vocal_2" is a vocals model from the
# Ultimate Vocal Remover project. This script never downloads it: the file has
# to be in models/ already.
MODEL_FILE = PROJECT_DIR / "models" / "Kim_Vocal_2.onnx"
MODEL_URL = (
    "https://github.com/TRvlvr/model_repo/releases/download/"
    "all_public_uvr_models/Kim_Vocal_2.onnx"
)

# The model was trained on 44.1 kHz audio and only understands that rate. A
# clip at any other rate is converted to it on the way in and back on the way
# out.
MODEL_SAMPLE_RATE = 44100

# The STFT the model was trained with. These are part of the model, not knobs:
# change them and its output turns to noise. The slice is much longer than the
# 2048 in common.py, which gives very fine detail in frequency. That is what
# lets the model tell a held musical note from the voice next to it.
MODEL_N_FFT = 7680
MODEL_HOP = 1024

# The model's voice comes out very slightly too quiet. Multiplying by this
# number brings it back to the right level. It was measured by the people who
# trained the model.
LEVEL_FIX = 1.009


def separate_mdx(audio, sr):
    """Split a mono signal into (foreground, background) with MDX-Net.

    audio: 1-D numpy array of samples.
    sr:    sample rate in Hz.

    Returns two arrays with the same length as `audio`: the voice, and
    everything else. Same inputs and outputs as separate() in separate.py.
    """
    if not MODEL_FILE.is_file():
        sys.exit(
            f"The MDX-Net model file is missing: {MODEL_FILE}\n"
            f"It is not downloaded automatically. It comes from: {MODEL_URL}"
        )

    # -----------------------------------------------------------------------
    # 1. Give the model the sample rate it expects: 44.1 kHz
    # -----------------------------------------------------------------------
    # Resample a working copy if the clip is at another rate. The files we
    # save later stay at the clip's own rate; this copy is only for the model.
    if sr != MODEL_SAMPLE_RATE:
        model_audio = librosa.resample(audio, orig_sr=sr, target_sr=MODEL_SAMPLE_RATE)
    else:
        model_audio = audio

    # -----------------------------------------------------------------------
    # 2. Load the model and read off the chunk size it wants
    # -----------------------------------------------------------------------
    # The .onnx file holds the network: its layers and its trained numbers.
    # onnxruntime is the engine that runs it. We ask for the plain CPU.
    model = onnxruntime.InferenceSession(
        str(MODEL_FILE), providers=["CPUExecutionProvider"]
    )
    input_name = model.get_inputs()[0].name

    # The model takes a fixed-size piece of spectrogram: `bins` frequency rows
    # (it only looks at the lowest ones, up to about 17.6 kHz) by `frames`
    # columns of time.
    _, _, bins, frames = model.get_inputs()[0].shape

    # How many samples of audio make exactly `frames` columns: about 6 seconds.
    chunk_samples = MODEL_HOP * (frames - 1)

    # The edges of a chunk come out worse than its middle, because the model
    # cannot see what comes before or after. So each chunk is cut a little
    # wider than needed, and this many samples are thrown away at both ends.
    edge = MODEL_N_FFT // 2
    step = chunk_samples - 2 * edge

    # Pad with silence so the first and last chunks have edges to throw away,
    # and so the length is a whole number of steps.
    num_steps = int(np.ceil(len(model_audio) / step))
    padded = np.zeros(edge + num_steps * step + edge, dtype=np.float32)
    padded[edge : edge + len(model_audio)] = model_audio

    # -----------------------------------------------------------------------
    # 3. Run the model one chunk at a time
    # -----------------------------------------------------------------------
    voice_chunks = []
    for start in range(0, num_steps * step, step):
        chunk = padded[start : start + chunk_samples]

        # STFT of the chunk: a grid of complex numbers, rows = frequencies,
        # columns = frames. Unlike REPET, we do not split off the phase. The
        # model works on the complex numbers themselves, so it can correct the
        # phase as well as the loudness.
        spectrum = librosa.stft(
            chunk, n_fft=MODEL_N_FFT, hop_length=MODEL_HOP, pad_mode="reflect"
        )

        # A neural network works on plain numbers, so each complex number is
        # split into its two parts (real and imaginary). The model was trained
        # on stereo, so it wants four grids: left real, left imaginary, right
        # real, right imaginary. Our clip is mono, so left and right are the
        # same. Only the lowest `bins` rows are passed in.
        low = spectrum[:bins]
        model_input = np.stack([low.real, low.imag, low.real, low.imag])
        model_input = model_input[None].astype(np.float32)  # add a batch dimension

        # The model answers with four grids of the same shape: its guess at
        # the spectrogram of the voice alone.
        model_output = model.run(None, {input_name: model_input})[0][0]

        # Put real and imaginary back together into complex numbers, and
        # average left and right to get one channel again. The rows above
        # `bins` stay zero: the model does not predict them.
        voice_spectrum = np.zeros_like(spectrum)
        voice_spectrum[:bins] = (
            model_output[0] + 1j * model_output[1]
            + model_output[2] + 1j * model_output[3]
        ) / 2

        # Back to sound with the inverse STFT, then drop the unreliable edges.
        voice = librosa.istft(
            voice_spectrum, n_fft=MODEL_N_FFT, hop_length=MODEL_HOP, length=chunk_samples
        )
        voice_chunks.append(voice[edge:-edge])

    # Join the chunks end to end and cut off the padding from step 2.
    foreground = np.concatenate(voice_chunks)[: len(model_audio)] * LEVEL_FIX

    # -----------------------------------------------------------------------
    # 4. Back to the clip's own format: original rate, original length
    # -----------------------------------------------------------------------
    # Undo the resampling from step 1, if there was any.
    if sr != MODEL_SAMPLE_RATE:
        foreground = librosa.resample(
            foreground, orig_sr=MODEL_SAMPLE_RATE, target_sr=sr
        )

    # Resampling can leave the result a sample or two off. Trim or pad so it
    # is exactly as long as the input and lines up with the original.
    foreground = librosa.util.fix_length(foreground, size=len(audio))

    # -----------------------------------------------------------------------
    # 5. The background is whatever is left
    # -----------------------------------------------------------------------
    # This model only predicts the voice. Taking the voice away from the
    # original leaves the music. Because of that, foreground + background is
    # exactly the original here, which is not true for REPET.
    background = audio - foreground

    return foreground, background
