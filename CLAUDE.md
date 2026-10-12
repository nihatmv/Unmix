# Unmix

A Python learning project: split a clip of speech over music into two layers
with classic DSP (no ML yet), and look at / listen to the result.

## The user is learning

- Keep code simple and flat: small functions, no classes or abstractions unless needed.
- Comment every signal-processing step in plain words (what it does and why),
  matching the style already in `src/`. Tunable numbers are named constants at
  the top of the file with a comment on what happens when you change them.
- Ask clarifying questions before starting a new phase.

## Commands

Always use the project venv (`.venv/bin/python`), never the system Python.

```
.venv/bin/pip install -r requirements.txt
.venv/bin/python src/visualize_original.py   [clip]   # -> output/original.png
.venv/bin/python src/separate.py             [clip]   # -> output/{original,foreground,background}.wav, source.txt
.venv/bin/python src/visualize_separation.py [clip]   # -> output/separation.png
.venv/bin/python src/serve.py                         # viewer at http://127.0.0.1:8765/viewer/
```

With no `[clip]`, the scripts use the first WAV/MP3 in `input/` alphabetically.
There is no test suite or linter.

## How it works

**Separation** (`separate_with_mask` in `src/separate.py`):

1. STFT of the mono signal, split into magnitude and phase.
2. `nn_filter` replaces each frame with the median of the frames that sound
   most like it (cosine similarity, ignoring frames closer than
   `SIMILARITY_GAP_SECONDS`). Music recurs so it survives the median; speech
   doesn't. The result, clipped to the original magnitude, is the estimated
   repeating background. Original minus that is the estimated foreground.
3. Two soft masks (values 0 to 1 per cell) are built with `librosa.util.softmask`,
   one per layer. `MARGIN_FOREGROUND` / `MARGIN_BACKGROUND` set how much stronger
   a layer must be to keep a cell; `MASK_POWER` sets how hard the decision is.
   The background mask is built from a second, more cautious estimate: the
   `BACKGROUND_PERCENTILE`-th percentile of the same look-alike frames instead
   of the median, which leaves out more of the voice. Rows below
   `VOICE_LOW_CUTOFF_HZ` skip the masks and go wholly to the background.
4. Each mask is multiplied onto the ORIGINAL magnitude, recombined with the
   mix's phase, and turned back into audio with the inverse STFT.

The masks use different margins, so foreground + background is close to the
original but not equal to it.

**Viewer** (`src/serve.py` + `viewer/index.html`):

1. On load the page calls `GET /api/state` and fills the dropdown with the
   clips in `input/`. If `output/` already has the three WAVs, it draws them.
2. **Separate** sends `POST /api/separate {"file": name}`. The server checks
   the name is really in `input/`, then runs `separate.py` on it as a
   subprocess and replies when it has finished (or with the error's last line).
   A lock allows one run at a time; a second request gets 409.
3. The page reloads the three WAVs from `output/` into wavesurfer and shows
   which clip they came from (`output/source.txt`).
4. Playback: each layer has its own player. Clicking any waveform seeks all
   three; "Play all synced" aligns them and starts them together.

## Layout

- `src/common.py` — shared paths, `N_FFT`/`HOP_LENGTH`, `find_input_file()`.
  Every script imports from here; keep STFT settings in this one place.
- `src/separate.py` — REPET-style separation (`librosa.decompose.nn_filter` + soft masks).
- `src/visualize_original.py`, `src/visualize_separation.py` — matplotlib plots.
- `src/serve.py` — stdlib HTTP server: serves the project root, plus
  `GET /api/state` and `POST /api/separate` (runs `separate.py` as a subprocess).
- `viewer/index.html` — single page, plain JS + wavesurfer.js 7.8.6 from unpkg. No frameworks, no build step.
- `input/`, `output/` — git-ignored apart from `.gitkeep`. Not a git repo yet.

## Things to preserve

- **`separate(audio, sr) -> (foreground, background)`** is the swap point for
  Demucs later. Mono float array in, two arrays of the same length out. Keep
  that signature. `separate_with_mask()` exists only so the plot can show the mask.
- **Audio is loaded mono at the native sample rate** (`sr=None, mono=True`). Don't resample.
- **The three WAVs in `output/` must be the same length**; the viewer's synced
  playback relies on it. `original.wav` is the mono signal the separation used.
- **Layer colours are shared** between the plots and the viewer: original
  `#2a78d6`, foreground `#eb6834`, background `#1baf7a`. Spectrograms use
  `magma` on a log-frequency axis; the three in `separation.png` share one dB scale.

## Gotchas

- **Intel Mac (x86_64), Python 3.10.** New `numba`/`llvmlite` releases have no
  wheel for this machine and fail to build from source. `requirements.txt`
  has `--only-binary numba,llvmlite` for that reason; keep it.
- **Port 8000 is used by another of the user's projects.** The viewer uses 8765.
  Don't kill whatever is on 8000.
- **`output/` holds one result at a time.** Every run overwrites it, and the
  PNGs are not redrawn by the viewer, so they can be from a different clip than
  the WAVs (`output/source.txt` names the clip the WAVs came from).
- **`visualize_separation.py` re-runs the separation**, so it is slow.
  Separation takes roughly 50 s per minute of audio.
- **Clips must be longer than about 4 s** (`2 * SIMILARITY_GAP_SECONDS`), or
  `separate` raises `ValueError`.
- The method only works when the background repeats and the foreground doesn't.
  A music-only clip gives a near-empty foreground; that is expected, not a bug.

## Checking changes

- Run the script and look at the output: read the PNG, or check the WAVs'
  length, sample rate and levels. Audio quality itself can only be judged by the user.
- For separation logic, use a synthetic mix with known sources (a repeating
  chord loop plus non-repeating tone bursts) and measure how much of each
  source lands in each layer. Write test clips to a scratch folder, not `input/`.
- For the viewer, start the server with `webbrowser.open` patched out (so no
  tab opens on the user's machine) and drive the page with headless Chrome at
  `/Applications/Google Chrome.app`.
- Testing through the real scripts overwrites `output/`; say so when it happens.
