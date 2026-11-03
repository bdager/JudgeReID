#!/usr/bin/env python3
"""Run statistics that survive an interrupt and add up across resumed runs.

* Parse counts are recorded per query in ``results.csv``, so they are
  recomputed from that file at the end, covering every run that appended to it.
* Wall-clock time and token usage are recorded nowhere else, so
  :class:`RunStats` keeps them in a ``run_stats.json`` file rewritten after
  every completed query: a hard kill loses at most the query in flight.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import time
from typing import Any, Dict, Iterable

import pandas as pd

logger = logging.getLogger(__name__)

RUN_STATS_FILE = "run_stats.json"

TOKEN_KEYS = (
    "num_api_calls",
    "total_prompt_tokens",
    "total_completion_tokens",
    "total_tokens",
)

# OpenAI usage attribute -> accumulator key
_USAGE_FIELDS = {
    "prompt_tokens": "total_prompt_tokens",
    "completion_tokens": "total_completion_tokens",
    "total_tokens": "total_tokens",
}

PARSE_STATUSES = ("ok", "ok_retry", "fallback")


def format_elapsed(secs: float) -> str:
    """Seconds as ``"1h 4m 12.30s"``."""
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}h {int(m)}m {s:.2f}s"


class RunStats:
    """Cumulative elapsed time and token usage for one ``output_dir``.

    Seeded from a previous ``run_stats.json`` in that directory, so the
    totals reported in ``metrics.json`` cover every run that contributed
    rows to ``results.csv`` — not just the last one.
    """

    def __init__(self, output_dir: str) -> None:
        self.path = os.path.join(output_dir, RUN_STATS_FILE)
        self._start = time.time()

        prior = self._load_prior()
        self.prior_elapsed_secs = float(prior.get("elapsed_secs") or 0.0)
        prior_tokens = prior.get("token_usage") or {}
        self.token_usage: Dict[str, int] = {
            k: int(prior_tokens.get(k) or 0) for k in TOKEN_KEYS
        }
        self.n_runs = int(prior.get("n_runs") or 0) + 1

        if self.prior_elapsed_secs or self.token_usage["num_api_calls"]:
            logger.info(
                "Carrying over totals from previous runs (%s): %s elapsed, %d API calls, %d tokens",
                RUN_STATS_FILE,
                format_elapsed(self.prior_elapsed_secs),
                self.token_usage["num_api_calls"],
                self.token_usage["total_tokens"],
            )

    def _load_prior(self) -> dict:
        if not os.path.exists(self.path):
            return {}
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            # Losing old totals is bad; refusing to start the run is worse
            logger.warning(
                "Could not read %s (%s) — restarting counters from zero",
                self.path,
                exc,
            )
            return {}

    @property
    def run_secs(self) -> float:
        """Wall-clock seconds spent in this invocation alone."""
        return time.time() - self._start

    @property
    def elapsed_secs(self) -> float:
        """Wall-clock seconds across every run that touched this output_dir."""
        return self.prior_elapsed_secs + self.run_secs

    def add_usage(self, usage: Any) -> bool:
        """Accumulate one API response's ``usage``.

        Returns False for the ``None`` handed back by a call that never
        succeeded, so ``num_api_calls`` counts billable calls only.
        """
        if usage is None:
            return False
        self.token_usage["num_api_calls"] += 1
        for attr, key in _USAGE_FIELDS.items():
            self.token_usage[key] += getattr(usage, attr, 0) or 0
        return True

    def add_usages(self, usages: Iterable[Any]) -> None:
        for usage in usages or []:
            self.add_usage(usage)

    def save(self) -> None:
        """Persist the cumulative totals.

        Called after every completed query, so it writes through a temp file
        and renames: a kill mid-write leaves the previous sidecar intact
        instead of a truncated one.
        """
        payload = {
            "elapsed_secs": round(self.elapsed_secs, 2),
            "elapsed_time": format_elapsed(self.elapsed_secs),
            "token_usage": dict(self.token_usage),
            "n_runs": self.n_runs,
        }
        tmp_path = f"{self.path}.tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=4)
            os.replace(tmp_path, self.path)
        except OSError as exc:
            logger.warning("Could not write %s: %s", self.path, exc)


def _one_row_per_query(df: pd.DataFrame) -> pd.DataFrame:
    """``load_checkpoint`` already drops superseded rows when requeueing with
    ``--retry_fallbacks``; dedup anyway so a hand-edited CSV can't inflate
    the counts."""
    return df.drop_duplicates(subset="query_path", keep="last")


def parse_stats_from_csv(results_csv_path: str) -> Dict[str, int]:
    """Count ``parse_status`` over ``results.csv``.

    Derived from the file rather than accumulated in memory so the counts
    cover every run that appended to it, matching the queries the reported
    metrics are actually computed over.
    """
    stats = {status: 0 for status in PARSE_STATUSES}
    if not os.path.exists(results_csv_path):
        return stats
    try:
        df = pd.read_csv(
            results_csv_path, usecols=["query_path", "parse_status"]
        )
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        logger.warning(
            "Could not derive parse_stats from %s: %s", results_csv_path, exc
        )
        return stats

    counts = _one_row_per_query(df)["parse_status"].astype(str).value_counts()
    for status, count in counts.items():
        stats[str(status)] = stats.get(str(status), 0) + int(count)
    return stats


def install_sigterm_handler() -> None:
    """Raise ``KeyboardInterrupt`` on SIGTERM.

    ``kill`` and job schedulers terminate the process silently by default,
    skipping the interrupt path that flushes stats and writes partial
    metrics; this routes them through the same handling as Ctrl-C.
    """

    def _handler(signum, _frame):
        raise KeyboardInterrupt(f"terminated by signal {signum}")

    try:
        signal.signal(signal.SIGTERM, _handler)
    except ValueError:
        pass  # not the main thread — nothing to install
