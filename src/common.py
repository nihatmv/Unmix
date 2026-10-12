"""Things every script in this project shares: folders, STFT settings, finding the clip."""

import sys
from pathlib import Path

# Folders, worked out from where this file lives so the scripts run the same
# no matter which directory you launch them from.
PROJECT_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_DIR / "input"
OUTPUT_DIR = PROJECT_DIR / "output"

AUDIO_EXTENSIONS = {".wav", ".mp3"}

# STFT settings. Try changing these and re-running to see the effect.
#
# N_FFT is how many samples go into each "slice" of audio that we analyse.
#   Bigger slice  -> finer detail in frequency, blurrier in time.
#   Smaller slice -> sharper in time, blurrier in frequency.
#   You can't have both; this trade-off is at the heart of all STFT work.
#
# HOP_LENGTH is how far we slide forward between slices. 512 with N_FFT=2048
#   means each slice overlaps the previous one by 75%, which gives a smooth
#   picture with no gaps.
N_FFT = 2048
HOP_LENGTH = 512


def find_input_file(clip=None):
    """Return the audio file to load: `clip` if given, or the first clip in input/.

    Scripts that take nothing but a clip on the command line can call this
    with no argument, and it reads the path from the command line itself.
    """
    if clip is None and len(sys.argv) > 1:
        clip = sys.argv[1]
    if clip is not None:
        path = Path(clip)
        if not path.is_file():
            sys.exit(f"File not found: {path}")
        return path

    clips = sorted(
        p for p in INPUT_DIR.iterdir() if p.suffix.lower() in AUDIO_EXTENSIONS
    )
    if not clips:
        sys.exit(f"No WAV or MP3 file found. Put one audio clip in {INPUT_DIR}/")
    return clips[0]
