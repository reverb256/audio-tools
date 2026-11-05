"""Small atomic-write helpers (tmp file + ``os.replace``)."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding=encoding) as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise
    return path


def atomic_write_json(path: Path, data) -> Path:
    return atomic_write_text(path, json.dumps(data, indent=2, sort_keys=False) + "\n")


def atomic_commit(tmp: Path, dst: Path) -> Path:
    """Move a finished temp artifact into place."""
    os.replace(tmp, dst)
    return dst


def fmt_hms(seconds: float) -> str:
    """mm:ss for chapters/tracklists (h:mm:ss past an hour)."""
    total = max(0, int(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
