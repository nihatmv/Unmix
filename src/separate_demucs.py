"""
A second way to split the clip: Demucs, a neural network trained on thousands
of songs to pull a mix apart into stems.

You don't run this file directly. Use the --method flag of separate.py:
    .venv/bin/python src/separate.py --method demucs
    .venv/bin/python src/separate.py --method both

It needs the extra packages in requirements-demucs.txt.

How it differs from the REPET method in separate.py: REPET only decides how
much of each spectrogram cell to keep, so hiding the voice also turns the music
down. Demucs has learned what voices and instruments sound like, so it can
rebuild the music underneath the voice.
"""

import librosa
import numpy as np
import torch
from demucs.apply import apply_model
from demucs.pretrained import get_model

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

# Which trained model to use. "htdemucs" is the standard Demucs v4 model. It
# splits a mix into four stems: drums, bass, other and vocals.
# "htdemucs_ft" is a fine-tuned version: a little cleaner, about 4x slower.
MODEL_NAME = "htdemucs"

# The model was trained on 44.1 kHz audio and only understands that rate. A
# clip at any other rate is converted to it on the way in and back on the way
# out.
MODEL_SAMPLE_RATE = 44100

# Long clips are processed in overlapping chunks that are then blended
# together. This is how much neighbouring chunks overlap, from 0 to 1.
#   Higher -> smoother joins between chunks, slower.
#   Lower  -> faster, but a join may become audible.
CHUNK_OVERLAP = 0.25

# The name of the stem that becomes our foreground.
FOREGROUND_STEM = "vocals"


def separate_demucs(audio, sr):
    """Split a mono signal into (foreground, background) with Demucs.

    audio: 1-D numpy array of samples.
    sr:    sample rate in Hz.

    Returns two arrays with the same length as `audio`: the voice, and
    everything else. Same inputs and outputs as separate() in separate.py.
    """
    # -----------------------------------------------------------------------
    # 1. Give Demucs the kind of audio it expects: 44.1 kHz stereo
    # -----------------------------------------------------------------------
    # Resample a working copy if the clip is at another rate. The files we
    # save later stay at the clip's own rate; this copy is only for the model.
    if sr != MODEL_SAMPLE_RATE:
        model_audio = librosa.resample(audio, orig_sr=sr, target_sr=MODEL_SAMPLE_RATE)
    else:
        model_audio = audio

    # The model wants two channels (left and right). Our clip is mono, so use
    # the same signal for both. The result has shape (2, samples).
    stereo = np.stack([model_audio, model_audio]).astype(np.float32)

    # -----------------------------------------------------------------------
    # 2. Bring the level to what the model saw in training
    # -----------------------------------------------------------------------
    # Shift the signal to be centred on zero and scale it to a standard
    # loudness. Demucs's own command-line tool does the same. We remember the
    # two numbers so the change can be undone on the way out.
    mean = stereo.mean()
    spread = stereo.std()
    if spread == 0:
        spread = 1.0  # a completely silent clip; avoid dividing by zero
    normalised = (stereo - mean) / spread

    # -----------------------------------------------------------------------
    # 3. Run the model
    # -----------------------------------------------------------------------
    # get_model loads the trained weights (downloading them the first time).
    # eval() puts the model in "use" mode rather than "training" mode.
    model = get_model(MODEL_NAME)
    model.eval()

    # The model takes a batch of clips, so add a leading "batch" dimension:
    # (2, samples) -> (1, 2, samples). torch.from_numpy turns the numpy array
    # into the tensor type PyTorch works with.
    batch = torch.from_numpy(normalised)[None]

    # no_grad switches off the bookkeeping that is only needed for training,
    # which saves time and memory. shifts=0 switches off a trick that runs the
    # model on randomly shifted copies and averages them; without it the same
    # clip always gives exactly the same result.
    with torch.no_grad():
        stems = apply_model(
            model,
            batch,
            device="cpu",
            shifts=0,
            split=True,
            overlap=CHUNK_OVERLAP,
            progress=True,
        )

    # The answer has shape (1, stems, 2, samples). Drop the batch dimension,
    # go back to numpy, and undo the level change from step 2.
    stems = stems[0].numpy() * spread + mean

    # -----------------------------------------------------------------------
    # 4. Four stems -> two layers
    # -----------------------------------------------------------------------
    # model.sources lists the stems in the order the model returns them.
    # The foreground is the vocals stem. The background is every other stem
    # (drums, bass, other) added back together.
    vocals_index = model.sources.index(FOREGROUND_STEM)
    foreground_stereo = stems[vocals_index]
    background_stereo = stems.sum(axis=0) - foreground_stereo

    # -----------------------------------------------------------------------
    # 5. Back to the clip's own format: mono, original rate, original length
    # -----------------------------------------------------------------------
    layers = []
    for layer_stereo in (foreground_stereo, background_stereo):
        # Average left and right to get one channel again.
        layer = layer_stereo.mean(axis=0)

        # Undo the resampling from step 1, if there was any.
        if sr != MODEL_SAMPLE_RATE:
            layer = librosa.resample(layer, orig_sr=MODEL_SAMPLE_RATE, target_sr=sr)

        # Resampling can leave the result a sample or two off. Trim or pad so
        # it is exactly as long as the input and lines up with the original.
        layer = librosa.util.fix_length(layer, size=len(audio))
        layers.append(layer)

    foreground, background = layers
    return foreground, background
