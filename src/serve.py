"""
Step 4: the interactive viewer.

Run from anywhere:
    .venv/bin/python src/serve.py

This starts a tiny web server on your own machine and opens the viewer in your
browser. In the viewer you pick a clip from input/, press Separate, and listen
to the layers. Press Ctrl+C here to stop the server.

Besides handing out the project's files, the server answers two requests from
the page:
    GET  /api/state      which clips are in input/, and what is in output/
    POST /api/separate   run src/separate.py on the chosen clip
"""

import json
import subprocess
import sys
import threading
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from common import AUDIO_EXTENSIONS, INPUT_DIR, OUTPUT_DIR, PROJECT_DIR

PORT = 8765
LAYERS = ["original.wav", "foreground.wav", "background.wav"]
SEPARATE_SCRIPT = Path(__file__).resolve().parent / "separate.py"

# Only one separation may run at a time: two at once would both write to the
# same files in output/.
separation_lock = threading.Lock()


def list_input_files():
    """Names of the audio clips in input/, in alphabetical order."""
    return sorted(
        p.name for p in INPUT_DIR.iterdir() if p.suffix.lower() in AUDIO_EXTENSIONS
    )


def current_state():
    """What the page needs to know to draw itself."""
    source_file = OUTPUT_DIR / "source.txt"
    return {
        "files": list_input_files(),
        "has_output": all((OUTPUT_DIR / name).is_file() for name in LAYERS),
        # Which clip the files in output/ were made from, if we know.
        "source": source_file.read_text().strip() if source_file.is_file() else None,
        "busy": separation_lock.locked(),
    }


class ViewerHandler(SimpleHTTPRequestHandler):
    """Serve the project's files, plus the two /api/ requests."""

    def end_headers(self):
        # Tell the browser not to keep copies. Without this it may keep playing
        # the old audio after a new separation.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def send_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/state":
            self.send_json(current_state())
        else:
            super().do_GET()

    def do_POST(self):
        if self.path != "/api/separate":
            self.send_json({"error": "Unknown request."}, status=404)
            return

        try:
            length = int(self.headers.get("Content-Length", 0))
            name = json.loads(self.rfile.read(length))["file"]
        except (ValueError, KeyError, TypeError):
            self.send_json({"error": "Expected a JSON body with a 'file'."}, status=400)
            return

        # Only accept a name that is really in input/. This also stops a request
        # from pointing the script at some other file on the computer.
        if name not in list_input_files():
            self.send_json({"error": f"{name} is not in input/."}, status=400)
            return

        if not separation_lock.acquire(blocking=False):
            self.send_json({"error": "A separation is already running."}, status=409)
            return
        try:
            # Run the same script you would run in the terminal, and wait for it.
            result = subprocess.run(
                [sys.executable, str(SEPARATE_SCRIPT), str(INPUT_DIR / name)],
                capture_output=True,
                text=True,
            )
        finally:
            separation_lock.release()

        if result.returncode != 0:
            # The last line of the error output is the useful part.
            lines = result.stderr.strip().splitlines()
            message = lines[-1] if lines else "Separation failed."
            self.send_json({"error": message}, status=500)
            return

        self.send_json(current_state())


def main():
    # "127.0.0.1" means only this computer can reach the server.
    handler = partial(ViewerHandler, directory=str(PROJECT_DIR))
    try:
        server = ThreadingHTTPServer(("127.0.0.1", PORT), handler)
    except OSError:
        sys.exit(f"Port {PORT} is already in use. Change PORT in src/serve.py.")

    url = f"http://127.0.0.1:{PORT}/viewer/"
    print(f"Viewer running at {url}  (Ctrl+C to stop)")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
