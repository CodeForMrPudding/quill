import os


class InstanceLock:
    """
    An exclusive lock on a file, held for as long as this object lives, so
    only one Quill process watches the PDF folder at a time. Running
    `quill watch` and a chat session (or two `quill watch`es) used to mean
    every new PDF was ingested and drafted twice, concurrently. The OS
    releases the lock automatically if the process dies.
    """

    def __init__(self, path: str):
        self.path = path
        self._fh = None

    def acquire(self) -> bool:
        """True if the lock was obtained; False if another process holds it."""
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        fh = open(self.path, "a+")
        try:
            if os.name == "nt":
                import msvcrt
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fh.close()
            return False
        self._fh = fh
        return True

    def release(self) -> None:
        if self._fh is None:
            return
        try:
            if os.name == "nt":
                import msvcrt
                self._fh.seek(0)
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        self._fh.close()
        self._fh = None
