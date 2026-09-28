from __future__ import annotations

import hashlib
import json
import logging
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from . import config

WS_RE = re.compile(r"\s+")


def setup_logging(name: str) -> logging.Logger:
    log_path = config.out("logs", f"{name}.log")
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_path)):
        h.setFormatter(fmt)
        root.addHandler(h)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    return logging.getLogger(name)


def normalize_spaces(text) -> str:
    return WS_RE.sub(" ", str(text or "")).strip()


def text_key(text: str) -> str:
    """Hash of lowercased alphanumeric content, for exact-duplicate detection."""
    t = re.sub(r"[^0-9a-z]+", " ", str(text).lower()).strip()
    return hashlib.sha1(t.encode()).hexdigest()


def truncate(text: str, max_chars: int) -> str:
    s = normalize_spaces(text)
    return s if len(s) <= max_chars else s[:max_chars].rstrip() + " ..."


def write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default, ensure_ascii=False)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


@contextmanager
def timed(log: logging.Logger, what: str):
    t0 = time.time()
    log.info("%s ...", what)
    yield
    log.info("%s done in %.1f s", what, time.time() - t0)
