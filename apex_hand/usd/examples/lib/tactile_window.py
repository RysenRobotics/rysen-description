#!/usr/bin/env python3
"""Drive the tactile pressure window from the simulator.

The window is the MuJoCo demo's ``TactilePanel``, but it cannot be created in
this process: Isaac Sim's virtualenv has no GUI toolkit for matplotlib (Agg
only), and importing Qt inside Kit crashes the simulator. So it runs in a
separate process (``tactile_panel_host.py``) under whatever Python does have a
toolkit, and this class streams the pad maps to it over localhost.

The wire format is a header plus raw little-endian float32, one frame per
update, so the window never has to import numpy data structures from here.

A failure to start the window is reported once and then ignored: the grasp run
is still useful without it, and a missing display should not stop it.
"""

import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path

_HOST = Path(__file__).resolve().parent / "tactile_panel_host.py"

# Send at most this often. Every physics step is 8.3 ms and the window is a
# matplotlib canvas; 20 Hz looks continuous and keeps the socket quiet.
SEND_INTERVAL = 0.05


def _candidate_pythons():
    """Interpreters that might have a GUI matplotlib, best first."""
    seen = []
    for candidate in (os.environ.get("TACTILE_WINDOW_PYTHON"),
                      "/usr/bin/python3", "python3",
                      shutil.which("python3")):
        if not candidate or candidate in seen:
            continue
        seen.append(candidate)
    return seen


def _has_gui_matplotlib(python):
    """True if that interpreter can import matplotlib and a Tk or Qt binding."""
    probe = (
        "import matplotlib;"
        "matplotlib.use('Agg');"
        "import importlib.util as u;"
        "ok = u.find_spec('tkinter') or u.find_spec('PySide6') or u.find_spec('PyQt5');"
        "raise SystemExit(0 if ok else 1)"
    )
    try:
        return subprocess.run([python, "-c", probe], timeout=25,
                              stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


class TactileWindow:
    """A pressure-map window in a child process, fed over localhost."""

    def __init__(self, side):
        self.side = side
        self._sock = None
        self._proc = None
        self._last = 0.0
        try:
            self._start()
        except Exception as error:      # noqa: BLE001 - the run must survive
            print(f"[tactile] pressure window unavailable: {error}",
                  file=sys.stderr)

    # ------------------------------------------------------------------ #
    def _start(self):
        python = next((p for p in _candidate_pythons()
                       if _has_gui_matplotlib(p)), None)
        if python is None:
            raise RuntimeError(
                "no Python with a GUI matplotlib found; set "
                "TACTILE_WINDOW_PYTHON or install matplotlib + tkinter")

        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        server.settimeout(60.0)
        port = server.getsockname()[1]

        self._proc = subprocess.Popen(
            [python, str(_HOST), str(port), self.side],
            stdout=subprocess.PIPE, text=True)
        try:
            self._sock, _ = server.accept()
        except socket.timeout:
            raise RuntimeError("the window process did not connect") from None
        finally:
            server.close()
        # Wait for its "ready" line so the first frames are not lost.
        line = self._proc.stdout.readline().strip()
        if "ready" not in line:
            raise RuntimeError(f"window process failed: {line or 'no output'}")
        print(f"[tactile] pressure window: {python} (pid {self._proc.pid})")

    # ------------------------------------------------------------------ #
    def update(self, tactiles):
        """Send one frame of ``tactiles.pressure_kpa`` if the interval elapsed."""
        if self._sock is None:
            return
        now = time.monotonic()
        if now - self._last < SEND_INTERVAL:
            return
        self._last = now
        maps = tactiles.pressure_kpa
        if not maps:
            return

        header = {"pads": [[name, int(image.shape[0]), int(image.shape[1])]
                           for name, image in maps.items()]}
        payload = json.dumps(header).encode()
        body = b"".join(
            image.astype("<f4").tobytes() for image in maps.values())
        try:
            self._sock.sendall(struct.pack("<II", len(payload), len(body)))
            self._sock.sendall(payload)
            self._sock.sendall(body)
        except OSError:
            self.close()

    def close(self):
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        if self._proc is not None:
            try:
                self._proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self._proc.terminate()
            self._proc = None
