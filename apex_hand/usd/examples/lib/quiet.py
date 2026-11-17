"""Drop the settings plugin's per-frame buffer warning from the console.

The plugin re-reads a viewport setting on every frame and logs

    [Warning] [carb.settings.plugin] getSharedArray: supplied array buffer
    doesn't fit the requested array

each time. It writes straight to the process streams, outside carb's Python
logging, so neither ``CARB_LOG_LEVEL`` nor ``carb.logging.set_level`` reaches
it. A headless run prints none -- there is no viewport to re-read the setting.

The only place left to drop it is the descriptor, so both streams are piped
through a line filter. Other output passes through untouched.

Call ``silence_settings_warnings()`` once, before Kit starts.
"""

import os
import threading

_NOISE = b"getSharedArray"


def _pump(read_fd, out_fd):
    """Forward whole lines that do not carry the warning, drop the rest."""
    pending = b""
    while True:
        chunk = os.read(read_fd, 65536)
        if not chunk:
            return
        pending += chunk
        while b"\n" in pending:
            line, pending = pending.split(b"\n", 1)
            if _NOISE not in line:
                os.write(out_fd, line + b"\n")


def silence_settings_warnings():
    """Redirect stdout and stderr through the filter."""
    for target in (1, 2):
        try:
            read_fd, write_fd = os.pipe()
            saved = os.dup(target)
            os.dup2(write_fd, target)
        except OSError:
            continue
        os.close(write_fd)
        threading.Thread(target=_pump, args=(read_fd, saved),
                         daemon=True).start()
