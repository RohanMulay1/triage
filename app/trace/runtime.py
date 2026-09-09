"""Process-lifetime safeguards for long live research collections."""
from __future__ import annotations

import os


class HostAwakeGuard:
    """Ask Windows to keep the host awake while a live collection is active.

    The request is process-scoped and Windows clears it if the process exits.
    Other platforms record that no native guard was available.
    """

    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.active = False
        self.platform = os.name

    def acquire(self) -> dict[str, object]:
        if not self.enabled or os.name != "nt":
            return self.status()
        import ctypes

        result = ctypes.windll.kernel32.SetThreadExecutionState(
            self.ES_CONTINUOUS | self.ES_SYSTEM_REQUIRED
        )
        if result == 0:
            raise RuntimeError("Windows refused the live-run host-awake request")
        self.active = True
        return self.status()

    def release(self) -> None:
        if not self.active:
            return
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(self.ES_CONTINUOUS)
        self.active = False

    def status(self) -> dict[str, object]:
        return {
            "requested": self.enabled,
            "active": self.active,
            "platform": self.platform,
        }
