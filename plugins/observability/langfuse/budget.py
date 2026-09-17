"""Profile-local deterministic head sampling and conservative observation reservations."""
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import logging
import math
from pathlib import Path
import sqlite3

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)


@lru_cache(maxsize=32)
def _settings(home: str):
    from hermes_cli.config_effective import load_user_config_effective
    config = load_user_config_effective(Path(home) / "config.yaml", fail_closed=True)
    raw = ((config or {}).get("observability") or {}).get("langfuse") or {}
    rate = raw.get("sample_rate")
    budget = raw.get("monthly_observation_budget")
    if rate is not None and (isinstance(rate, bool) or not isinstance(rate, (int, float))
                             or not math.isfinite(rate) or not 0 <= rate <= 1):
        raise ValueError("observability.langfuse.sample_rate must be between 0 and 1")
    if budget is not None and (isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0):
        raise ValueError("monthly_observation_budget must be a positive integer")
    return rate, budget


def configured_rate():
    return _settings(str(get_hermes_home()))[0]


def sampled(key):
    rate = configured_rate()
    if rate is None:
        return True
    bucket = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") / 2**64
    return bucket < rate


def reserve_observation():
    home = get_hermes_home()
    _, budget = _settings(str(home))
    if budget is None:
        return True
    directory = home / "logs"
    directory.mkdir(parents=True, exist_ok=True)
    month = datetime.now(timezone.utc).strftime("%Y-%m")
    # Reserve before export: a failed SDK call still consumes its reservation. This
    # intentionally overcounts, and cannot coordinate other hosts using this project.
    with sqlite3.connect(directory / "langfuse-budget.sqlite", timeout=5) as db:
        db.execute("CREATE TABLE IF NOT EXISTS observations (month TEXT PRIMARY KEY, count INTEGER NOT NULL)")
        db.execute("BEGIN IMMEDIATE")
        db.execute("INSERT OR IGNORE INTO observations VALUES (?, 0)", (month,))
        count = db.execute("SELECT count FROM observations WHERE month = ?", (month,)).fetchone()[0]
        if count >= budget:
            return False
        db.execute("UPDATE observations SET count = count + 1 WHERE month = ?", (month,))
    for fraction in (0.5, 0.75, 0.9):
        if count < budget * fraction <= count + 1:
            logger.warning("Langfuse local observation budget reached %d%% (%d/%d)", fraction * 100, count + 1, budget)
    return True


@lru_cache(maxsize=32)
def capture_mode(home):
    from hermes_cli.config_effective import load_user_config_effective
    config = load_user_config_effective(Path(home) / "config.yaml", fail_closed=True)
    value = ((config or {}).get("observability") or {}).get("langfuse", {}).get("capture")
    if value is not None and value not in {"metadata", "sanitized", "full"}:
        return "metadata"
    return value
