# Unmix

A small learning project: split a clip of speech-over-music into two layers
and look at the waves along the way. It has two methods: a classic DSP one
(REPET, no machine learning) and Demucs, a trained neural network.

## Setup

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

To use the Demucs method as well, install its extra packages (PyTorch and
Demucs, about 200 MB). The versions are pinned to ones that work on an Intel Mac:

```
.venv/bin/pip install -r requirements-demucs.txt
```

## Step 1: look at the original audio

Put one WAV or MP3 file in `input/`, then run:

```
.venv/bin/python src/visualize_original.py
```

The script prints the clip's duration, sample rate and sample count, and saves
`output/original.png` with the waveform on top and the spectrogram below.

## Step 2: split it into two layers

```
.venv/bin/python src/separate.py
```

This uses a REPET-style method: music repeats, speech doesn't. It saves
`output/foreground.wav` (the non-repeating part, hopefully the speech) and
`output/background.wav` (the repeating part, hopefully the music). Expect about
50 seconds of processing per minute of audio. The clip must be longer than
about 4 seconds.

The tuning knobs (`MARGIN_FOREGROUND`, `MARGIN_BACKGROUND`, `MASK_POWER`,
`SIMILARITY_GAP_SECONDS`, `BACKGROUND_PERCENTILE`, `VOICE_LOW_CUTOFF_HZ`) are at
the top of `src/separate.py`.

### Choosing the method

```
.venv/bin/python src/separate.py --method repet     # the default
.venv/bin/python src/separate.py --method demucs    # the neural network
.venv/bin/python src/separate.py --method both      # run both and compare
```

`repet` and `demucs` save to `output/` as above. Every run also saves the same
files into the method's own folder, `output/repet/` or `output/demucs/`, which
is where the viewer in step 4 reads them, so it can offer both versions.
`both` fills the two folders and draws `output/comparison.png`: the original
and both methods' foreground and background waveforms, stacked on one time axis
with the same y-scale. Every run prints how long each method took.

Demucs lives in `src/separate_demucs.py`. It expects 44.1 kHz stereo, so the
mono clip is duplicated into two channels (and resampled if needed) on the way
in, and turned back into mono at the clip's own rate on the way out.

## Step 3: look at the separation

```
.venv/bin/python src/visualize_separation.py
```

This runs the separation again and saves `output/separation.png`: the
waveforms and spectrograms of the original, foreground and background on a
shared time axis, plus the foreground mask.

## Step 4: pick a clip and listen in the browser

```
.venv/bin/python src/serve.py
```

This starts a small local server and opens `viewer/index.html`. There you can:

- choose any WAV or MP3 from `input/` and a **Method** (REPET, Demucs or
  Both), then press **Separate** to run the separation without touching the
  terminal ("Refresh list" picks up files you add while the page is open),
- pick which version to hear with **Listen to**: REPET or Demucs. Switching
  keeps the playhead where it is and keeps playing, so you can compare the two
  on the same moment. The status line names the clip and method you are hearing,
- see the three layers of that version as waveforms, each with its own Play
  button and Mute box,
- press "Play all synced" to start all three together. Clicking any waveform
  moves the playhead in all of them.

Press Ctrl+C to stop the server. The page loads wavesurfer.js from a CDN, so it
needs an internet connection. The viewer does not redraw the PNG plots; run the
step 1 and step 3 scripts for those.

## Using a specific file

The step 1-3 scripts use the first clip in `input/` unless you pass a path:

```
.venv/bin/python src/separate.py path/to/clip.wav
```
