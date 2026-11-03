"""
Prompt filling, response parsing, correction re-prompts and resume logic for
eval_vlm_rerank.py.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

CSV_FIELDS = [
    "query_path", "query_id", "n_candidates",
    "original_gt_rank", "reranked_gt_rank",
    "hit_at_1_before", "hit_at_1_after",
    "parse_status", "num_retries",
    "reranked_gallery_ids", "reranked_ranks",
]

SEP = "=" * 60


# ---------------------------------------------------------------------------
# Prompt placeholders
# ---------------------------------------------------------------------------
def build_vlm_placeholders(
    user_prompt_position: str = "before", label_images: bool = False,
) -> dict:
    """Values for the ``{KEY}`` placeholders in the prompt files.

    CANDIDATES_CONTEXT says "Below" when the instructions come before the
    images (``user_prompt_position='before'``) and "Above" otherwise.
    LABEL_NOTE announces the burned-in banners when ``label_images`` is set.
    """
    direction, verb = (
        ("Above", "see") if user_prompt_position == "after" else ("Below", "will see")
    )
    label_note = (
        " Each image has its label printed in a white banner at the top: "
        "\"Query\" for the Query person, and its number (1–{N}) for each candidate."
        if label_images else ""
    )
    return {
        "INPUT_FORMAT_RERANK": (
            "The user will show you a Query person crop and several numbered "
            "Candidate crops retrieved by a Re-ID system."
        ),
        "CANDIDATES_CONTEXT": (
            f"{direction} you {verb} a Query person followed by the numbered Candidates"
        ),
        "LABEL_NOTE": label_note,
    }


def fill_prompt_placeholders(text: str, placeholders: dict) -> str:
    """Replace every ``{KEY}`` in *text* with ``placeholders[KEY]``."""
    for key, value in placeholders.items():
        text = text.replace(f"{{{key}}}", value)
    return text


def build_example_line(n: int) -> str:
    """Example RANKING line: 1..n rotated by two, the same for every query, so it
    shows the format without suggesting an order."""
    perm = list(range(1, n + 1))
    if n > 2:
        perm = perm[2:] + perm[:2]
    nums = ", ".join(str(x) for x in perm)
    return f"Example for {n} candidates: RANKING: {nums}"


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------
_RANK_RE = re.compile(r"RANKING\s*:\s*([\d,\s]+)", re.IGNORECASE)


def parse_ranking(response: str, n: int) -> Optional[List[int]]:
    """Parse 'RANKING: 3, 1, 4, 2, 5' into a 0-indexed permutation.

    Returns None unless the line is a permutation of 1..n.
    """
    m = _RANK_RE.search(response)
    if not m:
        return None
    try:
        idx1 = [int(x.strip()) for x in m.group(1).split(",") if x.strip()]
    except ValueError:
        return None
    if sorted(idx1) != list(range(1, n + 1)):
        return None
    return [i - 1 for i in idx1]


def apply_reranking(candidates: List[dict], response: str) -> tuple[List[dict], str]:
    """Return (reranked_candidates, parse_status), with parse_status 'ok' or
    'fallback' (candidates left in their original order)."""
    order = parse_ranking(response, len(candidates))
    if order is not None:
        return [candidates[i] for i in order], "ok"
    return list(candidates), "fallback"


def diagnose_parse_failure(response: str, n: int) -> str:
    """Why *response* could not be parsed, in words the correction prompt can quote."""
    m = _RANK_RE.search(response)
    if not m:
        return "No RANKING line was found in your response."
    raw = m.group(1)
    try:
        idx1 = [int(x.strip()) for x in raw.split(",") if x.strip()]
    except ValueError:
        return f"The RANKING line contains non-integer values: '{raw.strip()}'."
    if len(idx1) != n:
        return f"Expected {n} candidate numbers but got {len(idx1)}: {idx1}."
    if sorted(idx1) != list(range(1, n + 1)):
        return (
            f"The ranking {idx1} is not a valid permutation of "
            f"1..{n} (check for duplicates or out-of-range values)."
        )
    return "The RANKING line could not be parsed."


def correction_message(correction_text: str, response: str, n: int) -> str:
    reason = diagnose_parse_failure(response, n)
    filled = correction_text.replace("{REASON}", reason).replace("{N}", str(n))
    return filled.replace("{EXAMPLE_LINE}", build_example_line(n))


async def run_attempt_with_retries_async(
    judge,
    messages: List[dict],
    candidates: List[dict],
    correction_text: str,
    max_retries: int,
    max_retries_delay: float,
    semaphore: asyncio.Semaphore,
) -> tuple[List[dict], str, int, str, List[Any]]:
    """Ask the judge for a ranking, re-prompting in the same conversation (up to
    *max_retries* times) while the answer is not a valid permutation.

    Returns (reranked_cands, parse_status, num_retries, response, usages).
    parse_status is 'ok', 'ok_retry' (fixed by a correction) or 'fallback'.
    Usages are returned rather than stored, since queries run concurrently.
    """
    n = len(candidates)
    usages: List[Any] = []

    async def _call(msgs: List[dict]) -> str:
        async with semaphore:
            text, usage = await judge.chat_async(msgs)
        usages.append(usage)
        return text

    try:
        response = await _call(messages)
    except Exception as exc:
        logger.warning("rerank error: %s — retrying…", exc)
        await asyncio.sleep(max_retries_delay)
        try:
            response = await _call(messages)
        except Exception as exc2:
            logger.error("rerank retry failed: %s", exc2)
            response = ""

    reranked_cands, parse_status = apply_reranking(candidates, response)

    conversation = messages + [{"role": "assistant", "content": response}]
    num_retries = 0
    while parse_status == "fallback" and num_retries < max_retries:
        num_retries += 1
        conversation.append(
            {"role": "user", "content": correction_message(correction_text, response, n)}
        )
        logger.info("Parse failed, correction retry %d/%d", num_retries, max_retries)
        try:
            response = await _call(conversation)
        except Exception as exc:
            logger.warning("Correction retry error: %s", exc)
            break
        conversation.append({"role": "assistant", "content": response})
        reranked_cands, parse_status = apply_reranking(candidates, response)

    if parse_status == "ok" and num_retries > 0:
        parse_status = "ok_retry"
    return reranked_cands, parse_status, num_retries, response, usages


# ---------------------------------------------------------------------------
# Paths and logs
# ---------------------------------------------------------------------------
def resolve_path(image_path: str, data_root: Optional[str]) -> str:
    p = Path(image_path)
    if p.is_absolute():
        return str(p)
    if data_root:
        return str(Path(data_root) / p)
    return str(p)


def remove_entries_from_log(path: str, qpaths_to_remove: set) -> None:
    """Rewrite a SEP-delimited log file without the entries of the given queries."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    kept = []
    for block in content.split(SEP):
        if not block.strip():
            continue
        first_line = block.strip().splitlines()[0]
        qp = first_line.replace("query:", "").strip() if first_line.startswith("query:") else None
        if qp not in qpaths_to_remove:
            kept.append(block)
    with open(path, "w", encoding="utf-8") as f:
        f.write(SEP.join(kept))
        if kept:
            f.write(SEP + "\n")


def first_rank(ranked_ids: List[int], qid: int) -> Optional[int]:
    for i, gid in enumerate(ranked_ids):
        if gid == qid:
            return i + 1
    return None


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------
def load_checkpoint(
    results_csv_path: str,
    max_retries: int,
    retry_fallbacks: bool,
    results_txt_path: str,
    unknowns_path: str,
) -> tuple[set, Dict[str, list]]:
    """Read an existing results.csv so a run can resume.

    Returns the query paths already done and their reranked gallery ids. With
    *retry_fallbacks*, unparsed queries that have retries left are removed from
    the CSV and the logs so they are processed again.
    """
    done_qpaths: set = set()
    restored_reranked: Dict[str, list] = {}
    if not os.path.exists(results_csv_path):
        return done_qpaths, restored_reranked

    # Read as text so rewriting the file keeps every value as it was
    existing = pd.read_csv(results_csv_path, dtype=str, keep_default_na=False)

    # To guarantee the files use the same columns:
    if list(existing.columns) != CSV_FIELDS:
        existing = existing.reindex(columns=CSV_FIELDS, fill_value="")
        existing.to_csv(results_csv_path, index=False)

    if retry_fallbacks:
        retry_qpaths = set()
        for _, row in existing.iterrows():
            try:
                prev_retries = int(row["num_retries"]) if pd.notna(row.get("num_retries")) else 0
            except (ValueError, TypeError):
                prev_retries = 0
            if str(row.get("parse_status", "")) == "fallback" and prev_retries < max_retries:
                retry_qpaths.add(str(row["query_path"]))
        if retry_qpaths:
            existing = existing[~existing["query_path"].astype(str).isin(retry_qpaths)]
            existing.to_csv(results_csv_path, index=False)
            for txt_path in (results_txt_path, unknowns_path):
                if os.path.exists(txt_path):
                    remove_entries_from_log(txt_path, retry_qpaths)
            logger.info(
                "Retry fallbacks: %d queries re-queued (%d kept)",
                len(retry_qpaths), len(existing),
            )

    done_qpaths = set(existing["query_path"].astype(str).tolist())
    for _, row in existing.iterrows():
        raw = str(row.get("reranked_gallery_ids", ""))
        if raw and raw != "nan":
            try:
                restored_reranked[str(row["query_path"])] = json.loads(raw)
            except ValueError:
                pass

    logger.info("Resuming: %d queries already done", len(done_qpaths))
    return done_qpaths, restored_reranked


# ---------------------------------------------------------------------------
# Result rows
# ---------------------------------------------------------------------------
def build_result_row(
    qpath: str,
    query_id: int,
    orig_ids: List[int],
    reranked_ids: List[int],
    reranked_ranks: List[int],
    n: int,
    parse_status: str,
    num_retries: int,
) -> dict:
    """One results.csv row.

    ``reranked_ranks`` is the same order as ``reranked_ids`` but as 1-indexed
    positions in the candidate pool. Unlike gallery ids, positions are unique,
    so they identify the exact candidate when a pool holds several crops of
    one identity (build_final_results.py relies on this).
    """
    orig_gt_rank = first_rank(orig_ids, query_id)
    new_gt_rank = first_rank(reranked_ids, query_id)
    return {
        "query_path": qpath,
        "query_id": query_id,
        "n_candidates": n,
        "original_gt_rank": orig_gt_rank,
        "reranked_gt_rank": new_gt_rank,
        "hit_at_1_before": int(orig_gt_rank == 1) if orig_gt_rank is not None else 0,
        "hit_at_1_after": int(new_gt_rank == 1) if new_gt_rank is not None else 0,
        "parse_status": parse_status,
        "num_retries": num_retries,
        "reranked_gallery_ids": json.dumps(reranked_ids),
        "reranked_ranks": json.dumps(reranked_ranks),
    }
