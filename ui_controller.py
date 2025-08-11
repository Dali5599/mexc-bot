
import threading, time, queue
from types import SimpleNamespace

class BotRunner:
    def __init__(self, entrypoint):
        self.entrypoint = entrypoint
        self.thread = None
        self.status = "stopped"
        self._stop = threading.Event()

    def start(self):
        if self.status == "running":
            return
        self._stop.clear()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        self.status = "running"

    def _run(self):
        # run the provided function; it should check for stop flag periodically
        try:
            self.entrypoint(self._stop)
        finally:
            self.status = "stopped"

    def stop(self):
        if self.status == "running":
            self._stop.set()
            if self.thread:
                self.thread.join(timeout=2)
            self.status = "stopped"
