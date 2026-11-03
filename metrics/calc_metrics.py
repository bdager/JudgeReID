#!/usr/bin/env python3
"""
Re-ranking metrics (CMC@k, mAP) for the VLM judge and the final rankings.
"""

import argparse
import json
import os
import sys
from collections import Counter
from typing import List, Optional, Sequence
from pathlib import Path
import numpy as np

# Run as a script, sys.path[0] is metrics/, so the metrics package isn't importable.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from metrics.run_stats import format_elapsed


# ---------------------------------------------------------------------------
# Re-ranking metrics
# ---------------------------------------------------------------------------
# Ranking metrics are stored at full precision. Rounding each subset and then
# rounding the subset average again can flip the second decimal of a
# published number (44.965 printed as 44.96); round only when printing.
def _keep(x, _ndigits=None) -> float:
    return float(x)


def _hit_at_k(ranked_ids: List[int], query_id: int, k: int) -> int:
    return int(query_id in ranked_ids[:k])


def _precision_at_k(ranked_ids: List[int], query_id: int, k: int) -> float:
    """Precision@k: fraction of the top-k ranked positions that are correct matches.
    Distinct from CMC@k, which is binary (>= 1 correct match in top-k). The denominator
    is min(k, len(ranked_ids)) so queries with fewer than k candidates available aren't
    penalized for slots that were never retrieved."""
    window = ranked_ids[:k]
    denom = len(window)
    if denom == 0:
        return 0.0
    n_hit = sum(1 for gid in window if gid == query_id)
    return n_hit / denom


def _average_precision(ranked_ids: List[int], query_id: int) -> float:
    """AP over a ranked list where relevance = (gallery_id == query_id)."""
    n_gt = sum(1 for gid in ranked_ids if gid == query_id)
    if n_gt == 0:
        return 0.0
    hits = 0
    sum_p = 0.0
    for i, gid in enumerate(ranked_ids):
        if gid == query_id:
            hits += 1
            sum_p += hits / (i + 1)
    return sum_p / n_gt


def _first_gt_rank(ranked_ids: List[int], query_id: int) -> Optional[int]:
    """1-indexed rank of the first GT item, or None if not present."""
    for i, gid in enumerate(ranked_ids):
        if gid == query_id:
            return i + 1
    return None


def extract_subset_from_path(path: str) -> str:
    """Mirror of metrics/evaluate_reid.py::extract_subset_from_path, duplicated
    here (rather than imported) so this module doesn't pull matplotlib/h5py
    into eval_vlm_rerank.py's hot path just for a path-parsing helper."""
    if "/test/" in path or "/train/" in path:
        parts = path.split("/")
        for i, part in enumerate(parts):
            if part in ["test", "train"] and i + 1 < len(parts):
                next_part = parts[i + 1]
                if next_part.startswith(("test_", "train_")):
                    return next_part
        return "unknown"
    else:
        filename = os.path.basename(path)
        parts = filename.split("_")
        if len(parts) >= 2:
            return f"{parts[0]}_{parts[1]}"
        return "unknown"


def _cmc_map(
    query_results: List[dict],
    ranked_key: str,
    topk_values: Sequence[int],
) -> dict:
    """Compute CMC@K, precision@K, and mAP for a list of query result dicts."""
    out: dict = {}
    for k in topk_values:
        hits = [
            _hit_at_k(r[ranked_key], r["query_id"], k) for r in query_results
        ]
        out[f"CMC@{k}"] = _keep(float(np.mean(hits)), 4) if hits else 0.0
        out[f"correct@{k}"] = f"{sum(hits)}/{len(hits)}"
        precisions = [
            _precision_at_k(r[ranked_key], r["query_id"], k) for r in query_results
        ]
        out[f"precision@{k}"] = _keep(float(np.mean(precisions)), 4) if precisions else 0.0
    aps = [
        _average_precision(r[ranked_key], r["query_id"]) for r in query_results
    ]
    out["mAP"] = _keep(float(np.mean(aps)), 4) if aps else 0.0
    out["n_queries"] = len(query_results)
    return out


def _subset_sem(values: Sequence[float]) -> float:
    """SEM over subsets. Reported metrics average subsets with equal weight, so
    the spread that matters is between subsets, not between queries."""
    n = len(values)
    if n < 2:
        return 0.0
    return float(np.std(values, ddof=1) / np.sqrt(n))


def _sem_block(per_subset_metrics: Sequence[dict], topk_values: Sequence[int]) -> dict:
    out = {
        f"CMC@{k}": _keep(_subset_sem([m[f"CMC@{k}"] for m in per_subset_metrics]), 4)
        for k in topk_values
    }
    out["mAP"] = _keep(_subset_sem([m["mAP"] for m in per_subset_metrics]), 4)
    return out


def _paired_sem_block(
    baselines: Sequence[dict], afters: Sequence[dict], topk_values: Sequence[int]
) -> dict:
    """SEM of the per-subset difference. Both conditions are scored on the same
    subsets, so pairing cancels the subset difficulty that dominates the
    unpaired SEM -- this is the spread to quote when comparing two rows."""
    out = {
        f"delta_CMC@{k}": _keep(
            _subset_sem([a[f"CMC@{k}"] - b[f"CMC@{k}"] for a, b in zip(afters, baselines)]),
            4,
        )
        for k in topk_values
    }
    out["delta_mAP"] = _keep(
        _subset_sem([a["mAP"] - b["mAP"] for a, b in zip(afters, baselines)]), 4
    )
    return out


def _cmc_map_per_subset(
    query_results: List[dict],
    ranked_key: str,
    topk_values: Sequence[int],
) -> dict:
    """Group by CHIRLA subset (test_N, parsed from query_path) and compute
    CMC@K/mAP per subset plus their simple (equal-weight-per-subset) average
    -- matches metrics/evaluate_reid.py's --per-subset convention. Each dict
    in query_results must include a 'query_path' key."""
    by_subset: dict = {}
    for r in query_results:
        subset = extract_subset_from_path(str(r["query_path"]))
        by_subset.setdefault(subset, []).append(r)

    per_subset = {
        subset: _cmc_map(rows, ranked_key, topk_values)
        for subset, rows in sorted(by_subset.items())
    }
    n = len(per_subset)
    avg: dict = {
        f"CMC@{k}": _keep(
            float(np.mean([m[f"CMC@{k}"] for m in per_subset.values()])), 4
        )
        for k in topk_values
    }
    avg["mAP"] = _keep(
        float(np.mean([m["mAP"] for m in per_subset.values()])), 4
    )
    return {
        "per_subset": per_subset,
        "subset_average": avg,
        "subset_sem": _sem_block(list(per_subset.values()), topk_values),
        "n_subsets": n,
    }


def rerank_metrics(
    all_query_results: List[dict],
    topk_values: Sequence[int] = (1, 5, 10),
    elapsed_secs: Optional[float] = None,
    token_usage: Optional[dict] = None,
    parse_stats: Optional[dict] = None,
    per_subset: bool = False,
) -> dict:
    """Compute four-tier re-ranking metrics.

    Parameters
    ----------
    all_query_results:
        One dict per query with keys:
          ``query_id``            – int ground-truth identity.
          ``original_ranked_ids`` – list[int] gallery IDs in original rank order.
          ``reranked_ids``        – list[int] after VLM, or ``None`` if not processed.
          ``gt_in_topk``          – bool, True when GT is present in the candidate list.
          ``query_path``          – str, required only when ``per_subset=True``
                                     (used to derive the CHIRLA test_N subset).
    topk_values:
        K values for CMC@K.
    per_subset:
        If True, additionally group queries by CHIRLA subset (test_1, test_2, …,
        parsed from ``query_path``) and report baseline/after-reranking CMC@K
        and mAP per subset, plus their simple average (equal weight per
        subset, matching metrics/evaluate_reid.py's --per-subset convention —
        this differs from the whole-set tiers above, which pool all queries
        together and are therefore weighted by subset size).
    """
    n_total = len(all_query_results)
    analyzable = [r for r in all_query_results if r["gt_in_topk"]]
    processed = [r for r in analyzable if r["reranked_ids"] is not None]
    n_analyzable = len(analyzable)
    n_processed = len(processed)
    n_skipped = n_total - n_analyzable

    metrics: dict = {
        "n_total_queries": n_total,
        "n_analyzable_queries": n_analyzable,
        "n_processed_queries": n_processed,
        "n_skipped_queries": n_skipped,
    }

    # Tier 1 — baseline, all queries, original ranking
    metrics["baseline_all_queries"] = _cmc_map(
        all_query_results, "original_ranked_ids", topk_values
    )

    # Tier 2 — after reranking, whole set
    # processed queries use reranked_ids; all others keep original
    def _effective(r: dict) -> List[int]:
        return (
            r["reranked_ids"]
            if r["reranked_ids"] is not None
            else r["original_ranked_ids"]
        )

    after_whole: dict = {}
    base_whole = metrics["baseline_all_queries"]
    for k in topk_values:
        hits = [
            _hit_at_k(_effective(r), r["query_id"], k)
            for r in all_query_results
        ]
        val = _keep(float(np.mean(hits)), 4) if hits else 0.0
        after_whole[f"CMC@{k}"] = val
        after_whole[f"delta_CMC@{k}"] = _keep(val - base_whole[f"CMC@{k}"], 4)
        precisions = [
            _precision_at_k(_effective(r), r["query_id"], k)
            for r in all_query_results
        ]
        prec_val = _keep(float(np.mean(precisions)), 4) if precisions else 0.0
        after_whole[f"precision@{k}"] = prec_val
        after_whole[f"delta_precision@{k}"] = _keep(
            prec_val - base_whole[f"precision@{k}"], 4
        )
    aps = [
        _average_precision(_effective(r), r["query_id"])
        for r in all_query_results
    ]
    after_whole["mAP"] = _keep(float(np.mean(aps)), 4) if aps else 0.0
    after_whole["delta_mAP"] = _keep(after_whole["mAP"] - base_whole["mAP"], 4)
    metrics["after_reranking_all_queries"] = after_whole

    # Tiers 3 & 4 — processed (analyzable + VLM-reranked) subset only
    if processed:
        base_sub = _cmc_map(processed, "original_ranked_ids", topk_values)
        after_sub = _cmc_map(processed, "reranked_ids", topk_values)

        after_sub_with_delta: dict = dict(after_sub)
        for k in topk_values:
            after_sub_with_delta[f"delta_CMC@{k}"] = _keep(
                after_sub[f"CMC@{k}"] - base_sub[f"CMC@{k}"], 4
            )
            after_sub_with_delta[f"delta_precision@{k}"] = _keep(
                after_sub[f"precision@{k}"] - base_sub[f"precision@{k}"], 4
            )
        after_sub_with_delta["delta_mAP"] = _keep(
            after_sub["mAP"] - base_sub["mAP"], 4
        )

        metrics["analyzed_subset_before"] = base_sub
        metrics["analyzed_subset_after"] = after_sub_with_delta

        # Tier 4 — rank-change breakdown
        rank_deltas: List[int] = []
        n_improved = n_degraded = n_unchanged = 0
        for r in processed:
            orig_rank = _first_gt_rank(r["original_ranked_ids"], r["query_id"])
            new_rank = _first_gt_rank(r["reranked_ids"], r["query_id"])
            if orig_rank is None or new_rank is None:
                continue
            delta = (
                orig_rank - new_rank
            )  # positive = rank improved (numerically lower)
            rank_deltas.append(delta)
            if delta > 0:
                n_improved += 1
            elif delta < 0:
                n_degraded += 1
            else:
                n_unchanged += 1

        metrics["rank_change_breakdown"] = {
            "n_improved": n_improved,
            "n_degraded": n_degraded,
            "n_unchanged": n_unchanged,
            "rank_delta_histogram": dict(sorted(Counter(rank_deltas).items())),
        }

    # Tier 5 (opt-in) — per-subset breakdown + simple (equal-weight) average
    if per_subset:
        by_subset: dict = {}
        for r in all_query_results:
            subset = extract_subset_from_path(str(r["query_path"]))
            by_subset.setdefault(subset, []).append(r)

        per_subset_out: dict = {}
        subset_baselines: List[dict] = []
        subset_afters: List[dict] = []
        for subset in sorted(by_subset):
            subset_results = by_subset[subset]
            sub_baseline = _cmc_map(subset_results, "original_ranked_ids", topk_values)
            sub_after: dict = {}
            for k in topk_values:
                hits = [
                    _hit_at_k(_effective(r), r["query_id"], k)
                    for r in subset_results
                ]
                sub_after[f"CMC@{k}"] = _keep(float(np.mean(hits)), 4) if hits else 0.0
                precisions = [
                    _precision_at_k(_effective(r), r["query_id"], k)
                    for r in subset_results
                ]
                sub_after[f"precision@{k}"] = (
                    _keep(float(np.mean(precisions)), 4) if precisions else 0.0
                )
            sub_aps = [
                _average_precision(_effective(r), r["query_id"])
                for r in subset_results
            ]
            sub_after["mAP"] = _keep(float(np.mean(sub_aps)), 4) if sub_aps else 0.0
            sub_after["n_queries"] = len(subset_results)

            per_subset_out[subset] = {
                "n_queries": len(subset_results),
                "baseline": sub_baseline,
                "after_reranking": sub_after,
            }
            subset_baselines.append(sub_baseline)
            subset_afters.append(sub_after)

        metrics["per_subset"] = per_subset_out

        if subset_baselines:
            avg_baseline = {
                f"CMC@{k}": _keep(
                    float(np.mean([b[f"CMC@{k}"] for b in subset_baselines])), 4
                )
                for k in topk_values
            }
            avg_baseline["mAP"] = _keep(
                float(np.mean([b["mAP"] for b in subset_baselines])), 4
            )
            avg_after = {
                f"CMC@{k}": _keep(
                    float(np.mean([a[f"CMC@{k}"] for a in subset_afters])), 4
                )
                for k in topk_values
            }
            avg_after["mAP"] = _keep(
                float(np.mean([a["mAP"] for a in subset_afters])), 4
            )
            for k in topk_values:
                avg_after[f"delta_CMC@{k}"] = _keep(
                    avg_after[f"CMC@{k}"] - avg_baseline[f"CMC@{k}"], 4
                )
            avg_after["delta_mAP"] = _keep(avg_after["mAP"] - avg_baseline["mAP"], 4)

            metrics["subset_average_baseline"] = avg_baseline
            metrics["subset_average_after_reranking"] = avg_after
            metrics["subset_sem_baseline"] = _sem_block(subset_baselines, topk_values)
            metrics["subset_sem_after_reranking"] = _sem_block(subset_afters, topk_values)
            metrics["subset_sem_delta"] = _paired_sem_block(
                subset_baselines, subset_afters, topk_values
            )
            metrics["n_subsets"] = len(subset_baselines)

    if elapsed_secs is not None:
        metrics["elapsed_time"] = format_elapsed(elapsed_secs)
        metrics["elapsed_secs"] = round(elapsed_secs, 2)

    if token_usage and token_usage.get("num_api_calls", 0) > 0:
        n = token_usage["num_api_calls"]
        metrics["token_usage"] = {
            "num_api_calls": n,
            "total_prompt_tokens": token_usage.get("total_prompt_tokens", 0),
            "total_completion_tokens": token_usage.get(
                "total_completion_tokens", 0
            ),
            "total_tokens": token_usage.get("total_tokens", 0),
            "avg_prompt_tokens": round(
                token_usage.get("total_prompt_tokens", 0) / n
            ),
            "avg_completion_tokens": round(
                token_usage.get("total_completion_tokens", 0) / n
            ),
            "avg_total_tokens": round(token_usage.get("total_tokens", 0) / n),
        }

    if parse_stats:
        metrics["parse_stats"] = parse_stats

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pandas as pd

    parser = argparse.ArgumentParser(
        description=(
            "Compute CMC/mAP metrics from a results CSV."
            "Supported formats (auto-detected from columns):"
            " - Rerank CSV  – columns include 'reranked_gallery_ids' and 'original_gt_rank'\n"
            " - Embeddings CSV – columns include 'gallery_id' and 'rank'"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results", required=True, help="Path to results CSV file."
    )
    parser.add_argument(
        "--topk",
        nargs="+",
        type=int,
        default=[1, 5, 10],
        help="K values for CMC@K (default: 1 5 10).",
    )
    parser.add_argument(
        "--output", default=None, help="Save metrics JSON to this path."
    )
    parser.add_argument(
        "--original_results",
        default=None,
        help="Path to the original embeddings CSV (long-format with query_path,"
        "query_id, gallery_id, rank). When provided together with a rerank "
        "results CSV, produces the full four-tier metrics identical to "
        "eval_vlm_rerank.py's metrics.json.",
    )
    parser.add_argument(
        "--topk_rerank",
        type=int,
        default=None,
        help="Top-K window used during reranking. Required with --original_results. If omitted, inferred from the n_candidates column in the rerank CSV.",
    )
    parser.add_argument(
        "--data_root",
        default=None,
        help="Prepended to relative image paths in --original_results "
        "(same as eval_vlm_rerank.py --data_root).",
    )
    parser.add_argument(
        "--tail_results",
        default=None,
        help="Path to an alternate long-format CSV (same columns as "
        "--original_results: query_path, query_id, gallery_id, rank) whose "
        "ranking supplies the tail (items beyond --topk_rerank) when merging "
        "with the VLM-reranked head, instead of --original_results' tail. "
        "Queries missing from this file fall back to --original_results' "
        "tail. If omitted, behavior is unchanged.",
    )
    parser.add_argument(
        "--eval-set",
        dest="eval_set",
        choices=["open", "closed", "both"],
        default="closed",
        help="'open' (default): score every query, matching prior behavior. "
        "'closed': keep only queries with query_id >= 0 (negative IDs mark "
        "unlabeled/distractor queries, per evaluate_reid.py's convention), "
        "matching its closed-set numbers exactly. 'both': report both, "
        "nested under 'open_set'/'closed_set'.",
    )
    parser.add_argument(
        "--per-subset",
        dest="per_subset",
        action="store_true",
        help="Also report CMC/mAP grouped by CHIRLA subset (test_1, test_2, "
        "…) plus their simple average, matching evaluate_reid.py's "
        "--per-subset convention. Applies to the four-tier rerank path "
        "(requires --original_results) and to the plain Embeddings CSV path.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.results)
    cols = set(df.columns)
    metrics: dict = {}

    if "reranked_gallery_ids" in cols and args.original_results:
        # ------------------------------------------------------------------
        # Full four-tier rerank metrics (same as eval_vlm_rerank.py)
        # ------------------------------------------------------------------
        orig_df = pd.read_csv(args.original_results)
        required = {"query_path", "query_id", "gallery_id", "rank"}
        if not required.issubset(orig_df.columns):
            print(
                f"original_results CSV missing columns: {required - set(orig_df.columns)}",
                file=sys.stderr,
            )
            sys.exit(1)

        def _resolve(p: str) -> str:
            pp = Path(p)
            if pp.is_absolute():
                return str(pp)
            if args.data_root:
                return str(Path(args.data_root) / pp)
            return str(pp)

        orig_df["query_path"] = orig_df["query_path"].apply(
            lambda p: _resolve(str(p))
        )

        tail_by_qpath: dict = {}
        if args.tail_results:
            no_rerank_df = pd.read_csv(args.tail_results)
            if not required.issubset(no_rerank_df.columns):
                print(
                    f"tail_results CSV missing columns: {required - set(no_rerank_df.columns)}",
                    file=sys.stderr,
                )
                sys.exit(1)
            no_rerank_df["query_path"] = no_rerank_df["query_path"].apply(
                lambda p: _resolve(str(p))
            )
            for qpath, grp in no_rerank_df.groupby("query_path"):
                grp = grp.sort_values("rank")
                tail_by_qpath[str(qpath)] = (
                    grp["gallery_id"].astype(int).tolist()
                )

        topk = args.topk_rerank
        if topk is None:
            if "n_candidates" in cols:
                topk = int(df["n_candidates"].iloc[0])
            else:
                print(
                    "Cannot infer topk_rerank: no n_candidates column. "
                    "Please provide --topk_rerank explicitly.",
                    file=sys.stderr,
                )
                sys.exit(1)

        reranked_by_qpath: dict = {}
        for _, row in df.iterrows():
            reranked_by_qpath[_resolve(str(row["query_path"]))] = json.loads(
                row["reranked_gallery_ids"]
            )

        all_query_results: list = []
        for qpath, grp in orig_df.groupby("query_path"):
            qpath = str(qpath)
            grp = grp.sort_values("rank")
            query_id = int(grp["query_id"].iloc[0])
            all_ids = grp["gallery_id"].astype(int).tolist()
            topk_ids = all_ids[:topk]
            gt_in_topk = query_id in topk_ids

            reranked_ids = None
            if qpath in reranked_by_qpath:
                tail_ids = tail_by_qpath.get(qpath, all_ids)[topk:]
                reranked_ids = reranked_by_qpath[qpath] + tail_ids

            all_query_results.append(
                {
                    "query_id": query_id,
                    "query_path": qpath,
                    "original_ranked_ids": all_ids,
                    "gt_in_topk": gt_in_topk,
                    "reranked_ids": reranked_ids,
                }
            )

        if args.eval_set == "open":
            metrics = rerank_metrics(
                all_query_results, topk_values=args.topk, per_subset=args.per_subset
            )
        elif args.eval_set == "closed":
            closed_results = [
                r for r in all_query_results if r["query_id"] >= 0
            ]
            metrics = rerank_metrics(
                closed_results, topk_values=args.topk, per_subset=args.per_subset
            )
        else:
            closed_results = [
                r for r in all_query_results if r["query_id"] >= 0
            ]
            metrics = {
                "open_set": rerank_metrics(
                    all_query_results, topk_values=args.topk, per_subset=args.per_subset
                ),
                "closed_set": rerank_metrics(
                    closed_results, topk_values=args.topk, per_subset=args.per_subset
                ),
            }

    elif "reranked_gallery_ids" in cols:
        # ------------------------------------------------------------------
        # Rerank CSV: one row per query, full reranked list available
        # ------------------------------------------------------------------
        query_results = [
            {
                "query_id": int(row["query_id"]),
                "ranked_ids": json.loads(row["reranked_gallery_ids"]),
            }
            for _, row in df.iterrows()
        ]

        if args.eval_set == "open":
            metrics.update(_cmc_map(query_results, "ranked_ids", args.topk))
        elif args.eval_set == "closed":
            closed_results = [r for r in query_results if r["query_id"] >= 0]
            metrics.update(_cmc_map(closed_results, "ranked_ids", args.topk))
        else:
            closed_results = [r for r in query_results if r["query_id"] >= 0]
            metrics["open_set"] = _cmc_map(
                query_results, "ranked_ids", args.topk
            )
            metrics["closed_set"] = _cmc_map(
                closed_results, "ranked_ids", args.topk
            )

        # Baseline from original_gt_rank
        # CMC@k: exact (first GT hit rank ≤ k)
        # mAP: approximate when multiple GTs exist in the candidate pool
        if "original_gt_rank" in cols:

            def _baseline(mask: "pd.Series") -> dict:
                n = int(mask.sum())
                b = {}
                for k in args.topk:
                    n_correct = int((df.loc[mask, "original_gt_rank"] <= k).sum())
                    b[f"baseline_CMC@{k}"] = round(n_correct / n, 4) if n else 0.0
                    b[f"baseline_correct@{k}"] = f"{n_correct}/{n}"
                b["baseline_mAP_approx"] = round(
                    float((1.0 / df.loc[mask, "original_gt_rank"]).mean()), 4
                )
                return b

            all_mask = pd.Series(True, index=df.index)
            closed_mask = df["query_id"] >= 0

            if args.eval_set == "open":
                metrics.update(_baseline(all_mask))
            elif args.eval_set == "closed":
                metrics.update(_baseline(closed_mask))
            else:
                metrics["open_set"].update(_baseline(all_mask))
                metrics["closed_set"].update(_baseline(closed_mask))

    elif "gallery_id" in cols and "rank" in cols:
        # ------------------------------------------------------------------
        # Embeddings CSV: one row per (query, gallery item), grouped by query
        # ------------------------------------------------------------------
        query_results = []
        for (qpath, query_id), group in df.groupby(["query_path", "query_id"]):
            ranked_ids = group.sort_values("rank")["gallery_id"].tolist()
            query_results.append(
                {
                    "query_id": int(query_id),
                    "query_path": str(qpath),
                    "ranked_ids": ranked_ids,
                }
            )

        if args.eval_set == "open":
            metrics.update(_cmc_map(query_results, "ranked_ids", args.topk))
            if args.per_subset:
                metrics.update(
                    _cmc_map_per_subset(query_results, "ranked_ids", args.topk)
                )
        elif args.eval_set == "closed":
            closed_results = [r for r in query_results if r["query_id"] >= 0]
            metrics.update(_cmc_map(closed_results, "ranked_ids", args.topk))
            if args.per_subset:
                metrics.update(
                    _cmc_map_per_subset(closed_results, "ranked_ids", args.topk)
                )
        else:
            closed_results = [r for r in query_results if r["query_id"] >= 0]
            metrics["open_set"] = _cmc_map(
                query_results, "ranked_ids", args.topk
            )
            metrics["closed_set"] = _cmc_map(
                closed_results, "ranked_ids", args.topk
            )
            if args.per_subset:
                metrics["open_set"].update(
                    _cmc_map_per_subset(query_results, "ranked_ids", args.topk)
                )
                metrics["closed_set"].update(
                    _cmc_map_per_subset(closed_results, "ranked_ids", args.topk)
                )

    else:
        print(
            "Unrecognized CSV format.\n"
            "  Rerank CSV must have column: 'reranked_gallery_ids'\n"
            "  Embeddings CSV must have columns: 'gallery_id', 'rank'",
            file=sys.stderr,
        )
        sys.exit(1)

    output_str = json.dumps(metrics, indent=2)
    print(output_str)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output_str)
