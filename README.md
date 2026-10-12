# Unmix

A small learning project: split a clip of speech-over-music into two layers
using classic DSP (no machine learning), and look at the waves along the way.

## Setup

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
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
40 seconds of processing per minute of audio. The clip must be longer than
about 4 seconds.

The tuning knobs (`MARGIN_FOREGROUND`, `MARGIN_BACKGROUND`, `MASK_POWER`,
`SIMILARITY_GAP_SECONDS`) are at the top of `src/separate.py`.

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

- choose any WAV or MP3 from `input/` and press **Separate** to run the
  separation without touching the terminal ("Refresh list" picks up files you
  add while the page is open),
- see the three layers from `output/` as waveforms, each with its own Play
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
