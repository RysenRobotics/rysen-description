#!/usr/bin/env python3
"""Draw the tactile pressure window in a process of its own.

Run with a Python that has a GUI matplotlib backend. Isaac Sim's virtualenv has
no toolkit at all (matplotlib is Agg-only there), and loading Qt inside the Kit
process crashes it, so the window cannot live in the simulator.

The window itself is the MuJoCo demo's ``TactilePanel`` -- this process only
receives pad maps from the simulator over a localhost socket and feeds them in,
which keeps one implementation of the layout and colour map for both models.

Arguments: ``port side``. Prints ``ready`` once the window is up; any traceback
goes to stderr so a failing window is visible in the simulator console.
"""

import json
import socket
import struct
import sys
from pathlib import Path

import numpy as np

# apex_hand/mjcf/examples/lib, from apex_hand/usd/examples/lib
_MJCF_LIB = Path(__file__).resolve().parents[3] / "mjcf" / "examples" / "lib"


class _Snapshot:
    """The attribute ``TactilePanel.update`` reads: pad name -> kPa image."""

    def __init__(self, maps):
        self.tactile_maps_kpa = maps


def _open_window_backend():
    """Pick a backend that can actually show a window, or exit."""
    import matplotlib

    errors = []
    for backend in ("TkAgg", "QtAgg", "Qt5Agg"):
        try:
            matplotlib.use(backend, force=True)
            import matplotlib.pyplot as plt

            probe = plt.figure()
            probe.canvas.draw()
            plt.close(probe)
            return plt
        except Exception as error:      # noqa: BLE001 - report and try the next
            errors.append(f"{backend}: {error}")
    print("tactile window: no GUI backend for matplotlib:\n  "
          + "\n  ".join(errors), file=sys.stderr)
    raise SystemExit(1)


def main():
    if len(sys.argv) < 3:
        raise SystemExit("usage: tactile_panel_host.py <port> <side>")
    port, side = int(sys.argv[1]), sys.argv[2]

    plt = _open_window_backend()
    sys.path.insert(0, str(_MJCF_LIB))
    from tactile_viewer import TactilePanel

    panel = TactilePanel(side)
    conn = socket.create_connection(("127.0.0.1", port))
    conn.settimeout(0.05)
    buffer = bytearray()
    print("tactile window: ready", flush=True)

    while plt.get_fignums():
        frame = None
        try:
            chunk = conn.recv(65536)
            if not chunk:
                break
            buffer.extend(chunk)
            frame = _parse(buffer)
        except socket.timeout:
            pass
        if frame is not None:
            panel.update(_Snapshot(frame))
        else:
            # No frame this tick: keep the Tk event loop alive so the window
            # still redraws and its close button works.
            plt.pause(0.001)

    plt.close("all")


def _parse(buffer):
    """Pop one complete frame out of ``buffer``; None if it is not all there."""
    if len(buffer) < 8:
        return None
    header_len, body_len = struct.unpack_from("<II", buffer, 0)
    total = 8 + header_len + body_len
    if len(buffer) < total:
        return None
    header = json.loads(bytes(buffer[8:8 + header_len]))
    offset = 8 + header_len
    maps = {}
    for name, rows, cols in header["pads"]:
        size = rows * cols * 4
        maps[name] = np.frombuffer(
            bytes(buffer[offset:offset + size]), dtype="<f4").reshape(rows, cols)
        offset += size
    del buffer[:total]
    return maps


if __name__ == "__main__":
    main()
