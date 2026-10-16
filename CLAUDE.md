# Unmix

A Python learning project: split a clip of speech over music into two layers
and look at / listen to the result. Phase 1 is classic DSP (REPET, no ML; git
tag `phase-1-repet`). Phase 2 adds two trained models as further methods:
Demucs, and MDX-Net, a vocals-only model that leaves less music in the voice.

## The user is learning

- Keep code simple and flat: small functions, no classes or abstractions unless needed.
- Comment every signal-processing step in plain words (what it does and why),
  matching the style already in `src/`. Tunable numbers are named constants at
  the top of the file with a comment on what happens when you change them.
- Ask clarifying questions before starting a new phase.

## Commands

Always use the project venv (`.venv/bin/python`), never the system Python.

```
.venv/bin/pip install -r requirements.txt          # REPET only
.venv/bin/pip install -r requirements-demucs.txt   # adds PyTorch + Demucs
.venv/bin/pip install -r requirements-mdx.txt      # adds onnxruntime (MDX-Net)
.venv/bin/python src/visualize_original.py   [clip]   # -> output/original.png
.venv/bin/python src/separate.py             [clip]   # -> output/{original,foreground,background}.wav, source.txt (and a copy in output/repet/)
.venv/bin/python src/separate.py --method demucs [clip]   # same files, made by Demucs
.venv/bin/python src/separate.py --method mdx    [clip]   # same files, made by MDX-Net
.venv/bin/python src/separate.py --method both   [clip]   # -> output/repet/, output/demucs/, output/comparison.png
.venv/bin/python src/visualize_comparison.py          # redraw comparison.png from those two folders
.venv/bin/python src/visualize_separation.py [clip]   # -> output/separation.png
.venv/bin/python src/serve.py                         # viewer at http://127.0.0.1:8765/viewer/
```

With no `[clip]`, the scripts use the first WAV/MP3 in `input/` alphabetically.
There is no test suite or linter.

## How it works

**REPET separation** (`separate_with_mask` in `src/separate.py`):

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

**Demucs separation** (`separate_demucs` in `src/separate_demucs.py`):

1. A working copy of the mono clip is resampled to 44.1 kHz if needed and
   duplicated into two channels, because the model only takes 44.1 kHz stereo.
2. The level is normalised (subtract mean, divide by standard deviation), as
   Demucs's own CLI does, and restored afterwards.
3. `htdemucs` runs on the CPU and returns four stems. Foreground = `vocals`;
   background = the other three added together.
4. Each layer is averaged back to mono, resampled back to the clip's rate if
   step 1 resampled, and trimmed/padded to the input's exact length.

**MDX-Net separation** (`separate_mdx` in `src/separate_mdx.py`):

1. A working copy of the mono clip is resampled to 44.1 kHz if needed.
2. The model file `models/Kim_Vocal_2.onnx` is loaded with onnxruntime (CPU).
   The clip is cut into chunks of about 6 s, each a little wider than needed
   so its unreliable edges can be thrown away.
3. Per chunk: STFT (`MODEL_N_FFT` 7680, `MODEL_HOP` 1024, fixed by the model),
   split into real and imaginary parts, duplicated to fake stereo. The model
   returns the voice's spectrogram, which is turned back into sound.
4. The chunks are joined, scaled by `LEVEL_FIX`, resampled back if step 1
   resampled, and trimmed/padded to the input's exact length.
5. Background = original minus voice, so here the two layers add up to the
   original exactly.

`separate.py --method` picks `repet` (default), `demucs`, `mdx` or `both`, and
prints how long each took. Every run writes each method's files to that
method's own folder, `output/repet/`, `output/demucs/` or `output/mdx/`, which
is where the viewer looks. A single method also writes them to `output/`
itself. `both` runs REPET and Demucs only, leaves the top-level WAVs alone and
draws `output/comparison.png` (five waveforms, one y-scale).

**Viewer** (`src/serve.py` + `viewer/index.html`):

1. On load the page calls `GET /api/state` and fills the dropdown with the
   clips in `input/`. The state lists three "versions", `repet`, `demucs` and
   `mdx`, one per folder in `output/`. A version with all three WAVs can be chosen
   under **Listen to**; the page draws the first one that exists.
2. **Separate** sends `POST /api/separate {"file": name, "method": method}`,
   where method is `repet`, `demucs`, `mdx` or `both` (the **Method** dropdown). The
   server checks the name is really in `input/`, then runs `separate.py` on it
   as a subprocess and replies when it has finished (or with the error's last
   line). A lock allows one run at a time; a second request gets 409.
3. The page loads the three WAVs of the version just made into wavesurfer
   (after `both` it stays on the version you were hearing) and shows which clip
   and method they are (`output/<method>/source.txt`).
4. Switching **Listen to** loads the chosen version's three WAVs, keeps the
   playhead and resumes whatever was playing.
5. Playback: each layer has its own player. Clicking any waveform seeks all
   three; "Play all synced" aligns them and starts them together.

## Layout

- `src/common.py` — shared paths, `N_FFT`/`HOP_LENGTH`, `find_input_file()`.
  Every script imports from here; keep STFT settings in this one place.
- `src/separate.py` — REPET-style separation (`librosa.decompose.nn_filter` + soft masks),
  plus `main()` with the `--method` flag, timing and saving for every method.
- `src/separate_demucs.py` — `separate_demucs(audio, sr)`. The only file that
  imports torch; `separate.py` imports it lazily so REPET works without it.
- `src/separate_mdx.py` — `separate_mdx(audio, sr)`. The only file that imports
  onnxruntime; also imported lazily. Needs no torch.
- `models/` — trained model files (`Kim_Vocal_2.onnx`, 67 MB). Git-ignored
  apart from `.gitkeep`.
- `src/visualize_comparison.py` — `save_comparison()` and a standalone redraw.
- `src/visualize_original.py`, `src/visualize_separation.py` — matplotlib plots.
- `src/serve.py` — stdlib HTTP server: serves the project root, plus
  `GET /api/state` and `POST /api/separate` (runs `separate.py --method ...` as a subprocess).
- `viewer/index.html` — single page, plain JS + wavesurfer.js 7.8.6 from unpkg. No frameworks, no build step.
- `input/`, `output/` — git-ignored apart from `.gitkeep`. Not a git repo yet.

## Things to preserve

- **`separate(audio, sr)`, `separate_demucs(audio, sr)` and `separate_mdx(audio, sr)`
  share one interface:** mono float array in, `(foreground, background)` of the
  same length out. Keep all three signatures. `separate_with_mask()` exists only so the plot can show the mask.
- **The REPET method is the frozen Phase 1 baseline.** Don't change its logic
  or settings unless the user asks.
- **Audio is loaded mono at the native sample rate** (`sr=None, mono=True`). Don't
  resample what gets saved. Demucs and MDX-Net resample internally only.
- **The three WAVs of a version must be the same length**; the viewer's synced
  playback relies on it. `original.wav` is the mono signal the separation used.
- **Layer colours are shared** between the plots and the viewer: original
  `#2a78d6`, foreground `#eb6834`, background `#1baf7a`. Spectrograms use
  `magma` on a log-frequency axis; the three in `separation.png` share one dB scale.

## Gotchas

- **Intel Mac (x86_64), Python 3.10.** New `numba`/`llvmlite` releases have no
  wheel for this machine and fail to build from source. `requirements.txt`
  has `--only-binary numba,llvmlite` for that reason; keep it.
- **The Demucs pins in `requirements-demucs.txt` are all forced by this machine:**
  `torch==2.2.2` / `torchaudio==2.2.2` (last Intel-Mac wheels), `numpy<2`
  (that torch can't talk to NumPy 2), `demucs==4.0.1` (4.1.0 needs `sphn`,
  which has no Intel-Mac wheel). Don't upgrade any of them.
- **REPET's output depends on the NumPy version.** In `connectivity` mode
  librosa's `recurrence_matrix` keeps an arbitrary 62 of each frame's ~400
  nearest look-alikes (it sorts a list of all-ones), and the order differs
  between NumPy 1 and 2. Results are repeatable within one environment and
  score the same on known-source mixes, but are not sample-identical across
  the two. Picking the true 62 nearest (`mode="distance"`) measures worse.
- **Demucs loads its weights from `~/.cache/torch/hub/checkpoints/`** and
  downloads them (about 80 MB) if they are missing. It takes roughly 80 s per
  minute of audio on this CPU, including about 10 s to load.
- **Never download or install without asking the user first.** That includes
  model files, packages, and code that downloads as a side effect.
  `separate_mdx.py` does not download its model: if `models/Kim_Vocal_2.onnx`
  is missing it exits and names the source URL.
- **onnxruntime was installed without its declared extras** (`coloredlogs`,
  `flatbuffers`, `protobuf`), so `pip check` complains. Running the model does
  not need them.
- **MDX-Net takes about 1 s per second of audio on this CPU**, a little faster
  than Demucs. Mel-Band RoFormer measured cleaner still on one test mix but
  was far too slow here (around 15 s per second of audio); don't suggest it.
- **Port 8000 is used by another of the user's projects.** The viewer uses 8765.
  Don't kill whatever is on 8000.
- **`output/` holds one result per method.** Every run overwrites the folders
  of the methods it ran, so `output/repet/`, `output/demucs/` and `output/mdx/`
  can hold different clips (each folder's `source.txt` names its clip; the viewer's
  status line shows it, and `visualize_comparison.py` refuses to compare two
  different clips). The PNGs are not redrawn by the viewer, so they can be from
  a different clip than the WAVs.
- **`visualize_separation.py` re-runs the separation**, so it is slow.
  Separation takes roughly 50 s per minute of audio.
- **Clips must be longer than about 4 s** (`2 * SIMILARITY_GAP_SECONDS`), or
  `separate` raises `ValueError`.
- REPET only works when the background repeats and the foreground doesn't.
  A music-only clip gives a near-empty foreground; that is expected, not a bug.
- Demucs and MDX-Net put any human voice in the foreground, so singing or
  humming in the music ends up with the speech.

## Checking changes

- Run the script and look at the output: read the PNG, or check the WAVs'
  length, sample rate and levels. Audio quality itself can only be judged by the user.
- For separation logic, use a synthetic mix with known sources (a repeating
  chord loop plus non-repeating tone bursts) and measure how much of each
  source lands in each layer. Write test clips to a scratch folder, not `input/`.
- The same known-source mixes work for Demucs; it should beat REPET clearly
  (around 10 dB or more better on the voice layer).
- For the trained models, mixes of real speech plus real music tell more than
  tone bursts. On four such mixes at equal loudness, voice versus leaked music
  measured about 17 dB for Demucs and about 24 dB for MDX-Net. Changing
  Demucs's `shifts`, overlap or model (`htdemucs_ft`) moved it by 1.5 dB at most.
- The user often has their own `serve.py` running on 8765. Don't kill it; test
  on another port (set `serve.PORT` before `serve.main()`). A server started
  before a code change must be restarted by the user to pick it up.
- For the viewer, start the server with `webbrowser.open` patched out (so no
  tab opens on the user's machine) and drive the page with headless Chrome at
  `/Applications/Google Chrome.app`.
- Testing through the real scripts overwrites `output/`; say so when it happens.
