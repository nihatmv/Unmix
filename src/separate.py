"""
Step 2: split the clip into two layers with a REPET-style method (no ML).

Run from anywhere:
    .venv/bin/python src/separate.py            # first clip in input/
    .venv/bin/python src/separate.py my.wav     # a specific file

It saves:
    output/original.wav     the clip as loaded (mono), for comparison
    output/foreground.wav   the non-repeating part (hopefully the speech)
    output/background.wav   the repeating part     (hopefully the music)

The idea in one sentence: music repeats itself, speech doesn't. If a patch of
the spectrogram looks the same every time a similar moment comes around, it is
probably music; whatever sticks out above that is probably the voice.
"""

import sys

import librosa
import numpy as np
import soundfile as sf

from common import HOP_LENGTH, N_FFT, OUTPUT_DIR, find_input_file

# ---------------------------------------------------------------------------
# Settings. Try changing these and listening to what happens.
# ---------------------------------------------------------------------------

# When looking for "moments that sound like this one", ignore anything closer
# than this many seconds. Without it, each moment's best matches would just be
# its immediate neighbours, which contain the same syllable of speech, and the
# voice would be mistaken for something that repeats.
SIMILARITY_GAP_SECONDS = 2.0

# How strict each mask is. A margin of 1 means "whichever layer is stronger
# wins the cell". A margin above 1 means the layer has to be that many times
# stronger than the other one before it gets to keep the cell.
#   Higher -> cleaner layer, less of the other one leaking in, but more of the
#             layer itself gets thrown away (it starts to sound thin or watery).
#   Lower  -> fuller sound, more leakage.
# The foreground margin is higher because music leaking into the voice is more
# annoying than a little voice leaking into the music.
MARGIN_BACKGROUND = 2
MARGIN_FOREGROUND = 10

# How "soft" the masks are. 1 gives gentle, blurry decisions; larger values push
# every cell towards fully kept or fully removed. 2 is the usual middle ground.
MASK_POWER = 2

# How cautious the background's guess at the music is, from 0 to 100.
# The voice only ever ADDS energy on top of the music, so among a set of
# look-alike moments the quieter ones are the ones with less voice in them.
# 50 is the plain median (the middle value). A lower number leans towards the
# quieter look-alikes, which contain less voice.
#   Lower  -> less voice left in the background, but the music gets quieter
#             and more hollow while someone is talking.
#   Higher -> fuller music, more of a faint "ghost" voice left behind.
# Only the background uses this; the foreground still uses the median.
BACKGROUND_PERCENTILE = 25

# A speaking voice has almost nothing below this frequency, so everything
# under it (bass, kick drum, rumble) is handed straight to the background.
#   Higher -> more bass kept whole in the music, but a deep voice starts to
#             lose its lowest notes (to the background).
#   Lower  -> safer for very deep voices, more holes in the bass.
#   0      -> switches this off.
VOICE_LOW_CUTOFF_HZ = 90


def separate(audio, sr):
    """Split a mono signal into (foreground, background).

    audio: 1-D numpy array of samples.
    sr:    sample rate in Hz.

    Returns two arrays with the same length as `audio`: the non-repeating layer
    (speech) and the repeating layer (music).

    This is the function to replace when swapping in another separator.
    """
    foreground, background, _ = separate_with_mask(audio, sr)
    return foreground, background


def separate_with_mask(audio, sr):
    """Do the separation and also hand back the foreground mask, for plotting.

    Returns (foreground, background, mask_foreground). The mask is a grid of
    numbers from 0 to 1, one per spectrogram cell (rows = frequencies,
    columns = frames).
    """
    # -----------------------------------------------------------------------
    # 1. STFT: split into magnitude and phase
    # -----------------------------------------------------------------------
    # The STFT turns the audio into a grid: rows are frequencies, columns are
    # short moments in time ("frames"). Each cell is a complex number, which we
    # split into two parts:
    #   magnitude -> how strong that frequency is at that moment
    #   phase     -> where in its cycle the wave is
    # All the separation work happens on the magnitude. The phase is set aside
    # and reused untouched at the end to turn the grids back into sound.
    stft = librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH)
    magnitude, phase = librosa.magphase(stft)

    # -----------------------------------------------------------------------
    # 2. Estimate the repeating background
    # -----------------------------------------------------------------------
    # Convert the "ignore nearby moments" gap from seconds to frames.
    gap_frames = int(
        librosa.time_to_frames(SIMILARITY_GAP_SECONDS, sr=sr, hop_length=HOP_LENGTH)
    )
    num_frames = magnitude.shape[1]
    if gap_frames >= (num_frames - 1) // 2:
        raise ValueError(
            f"Clip is too short to look for repetition: it needs to be longer "
            f"than about {2 * SIMILARITY_GAP_SECONDS:.0f} seconds."
        )

    # For every frame, find the other frames in the clip whose spectrum looks
    # most like it ("nearest neighbours"). The answer is a table saying which
    # frames are look-alikes of which. It is the slow part, so we work it out
    # once and use it twice below.
    #
    #   metric="cosine" -> compare the SHAPE of two spectra and ignore their
    #                      overall loudness, so a quiet chorus still matches a
    #                      loud one.
    look_alikes = librosa.segment.recurrence_matrix(
        magnitude,
        metric="cosine",
        width=gap_frames,
        mode="connectivity",
        sparse=True,
    )

    # nn_filter replaces each cell with the median across its look-alike
    # frames. The median ignores outliers. The music is there in most of the
    # look-alike frames, so it survives. The voice says something different in
    # each one, so it is the outlier and gets dropped.
    #
    # The result is a new magnitude grid: our guess at the music on its own.
    repeating = librosa.decompose.nn_filter(
        magnitude, rec=look_alikes, aggregate=np.median
    )

    # The background is one part of the mix, so it can never be louder than the
    # mix itself. Wherever the guess overshoots, clip it down to the original.
    repeating = np.minimum(magnitude, repeating)

    # Whatever is left once the repeating part is taken away is our guess at the
    # voice: energy that showed up here but not in the look-alike frames.
    non_repeating = magnitude - repeating

    # A second, more cautious guess at the music, used only for the background.
    # When the voice is loud, look-alike frames are often picked because the
    # VOICE sounds alike in them, and then the median still has some voice in
    # it. Taking a lower percentile (a value nearer the quietest look-alike
    # than the middle one) leaves most of that voice out.
    def low_percentile(values, axis):
        return np.percentile(values, BACKGROUND_PERCENTILE, axis=axis)

    repeating_cautious = librosa.decompose.nn_filter(
        magnitude, rec=look_alikes, aggregate=low_percentile
    )
    repeating_cautious = np.minimum(magnitude, repeating_cautious)
    non_repeating_cautious = magnitude - repeating_cautious

    # -----------------------------------------------------------------------
    # 3. Build the soft masks
    # -----------------------------------------------------------------------
    # A mask is a grid the same shape as the spectrogram, holding a number from
    # 0 to 1 in every cell. It answers "what fraction of this cell belongs to
    # this layer?": 1 = keep it all, 0 = remove it all, 0.3 = keep 30%.
    #
    # A "soft" mask allows in-between values. A hard mask (only 0 or 1) would
    # switch cells on and off abruptly, which you hear as gurgling "musical
    # noise". Sharing a cell between layers sounds much smoother.
    #
    # softmask(X, X_ref) compares two grids cell by cell and returns
    #       X^p / (X^p + X_ref^p)
    # which is close to 1 where X is much bigger than X_ref, close to 0 where
    # it is much smaller, and 0.5 where they are equal.

    # Background mask: close to 1 where the repeating part dominates the cell.
    # Multiplying the competitor by the margin makes it look bigger than it is,
    # so the music only keeps cells where it clearly wins.
    # It uses the cautious guess, so less voice gets through.
    mask_background = librosa.util.softmask(
        repeating_cautious,
        MARGIN_BACKGROUND * non_repeating_cautious,
        power=MASK_POWER,
    )

    # Foreground mask: the same comparison the other way round. Close to 1 where
    # the non-repeating part dominates, i.e. where the voice stands out above
    # the music.
    mask_foreground = librosa.util.softmask(
        non_repeating,
        MARGIN_FOREGROUND * repeating,
        power=MASK_POWER,
    )

    # Below the lowest notes of a speaking voice there is only music, so no
    # guessing is needed there: those rows go wholly to the background. Without
    # this, the bass drops out of the music every time someone speaks, because
    # the look-alike frames were matched on the voice and their bass differs.
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    below_voice = frequencies < VOICE_LOW_CUTOFF_HZ
    mask_background[below_voice, :] = 1.0
    mask_foreground[below_voice, :] = 0.0

    # -----------------------------------------------------------------------
    # 4. Apply the masks
    # -----------------------------------------------------------------------
    # Multiply the ORIGINAL magnitude by each mask, cell by cell. Each mask acts
    # like a stencil laid over the spectrogram: cells near 1 pass through, cells
    # near 0 are silenced.
    #
    # We mask the original rather than using `repeating` / `non_repeating`
    # directly because those are rough estimates (the median blurs detail),
    # while the original holds the real, sharp sound. The estimates only decide
    # WHERE each layer lives; the sound itself comes from the mix.
    #
    # Because the two masks use different margins they don't add up to exactly
    # 1, so foreground + background is close to the original but not identical.
    magnitude_foreground = mask_foreground * magnitude
    magnitude_background = mask_background * magnitude

    # -----------------------------------------------------------------------
    # 5. Back to sound with the inverse STFT
    # -----------------------------------------------------------------------
    # Put the mix's phase back onto each masked magnitude to get complex STFTs
    # again, then run the inverse STFT, which overlaps and adds the frames back
    # into one waveform. We never worked out a separate phase for each layer,
    # so both borrow the mix's phase. It is not exactly right, but it is close
    # enough wherever one layer dominates a cell.
    #
    # length=len(audio) trims the result to the exact length of the input.
    foreground = librosa.istft(
        magnitude_foreground * phase, hop_length=HOP_LENGTH, length=len(audio)
    )
    background = librosa.istft(
        magnitude_background * phase, hop_length=HOP_LENGTH, length=len(audio)
    )

    return foreground, background, mask_foreground


def main():
    audio_path = find_input_file()

    # Same loading as in step 1: mono, at the file's own sample rate.
    audio, sr = librosa.load(audio_path, sr=None, mono=True)
    print(f"File:         {audio_path.name}")
    print(f"Duration:     {len(audio) / sr:.2f} seconds at {sr} Hz")
    print("Separating... (this can take a little while on long clips)")

    try:
        foreground, background = separate(audio, sr)
    except ValueError as error:
        sys.exit(str(error))

    # Save both layers as WAV at the original sample rate so they play back at
    # the right speed and pitch.
    OUTPUT_DIR.mkdir(exist_ok=True)
    #
    # The original is saved too, as the exact mono signal the separation
    # worked on. All three files then have the same length and line up sample
    # for sample, which the viewer relies on to play them in sync.
    original_path = OUTPUT_DIR / "original.wav"
    foreground_path = OUTPUT_DIR / "foreground.wav"
    background_path = OUTPUT_DIR / "background.wav"
    sf.write(original_path, audio, sr)
    sf.write(foreground_path, foreground, sr)
    sf.write(background_path, background, sr)
    # Note which clip these files came from, so the viewer can show its name.
    (OUTPUT_DIR / "source.txt").write_text(audio_path.name)
    print(f"Saved:        {original_path}")
    print(f"Saved:        {foreground_path}")
    print(f"Saved:        {background_path}")


if __name__ == "__main__":
    main()
