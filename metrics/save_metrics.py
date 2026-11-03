#!/usr/bin/env python3
"""
Logging and saving of the VLM judge's metrics (metrics.json).
"""

import json
import logging
import os

logger = logging.getLogger(__name__)


def log_and_save_rerank_metrics(
    output_dir: str,
    metrics: dict,
    sep_width: int = 60,
) -> None:
    """Log a re-ranking metrics summary and persist to ``metrics.json``."""
    sep = "=" * sep_width
    logger.info(sep)
    logger.info("RE-RANKING RESULTS SUMMARY")
    logger.info("  Total queries      : %d", metrics.get("n_total_queries",      0))
    logger.info("  Analyzable (GT@K)  : %d", metrics.get("n_analyzable_queries", 0))
    logger.info("  Processed by VLM   : %d", metrics.get("n_processed_queries",  0))
    logger.info("  Skipped (GT miss)  : %d", metrics.get("n_skipped_queries",    0))

    def _fmt_tier(label: str, before: dict, after: dict) -> None:
        logger.info("")
        logger.info("  [%s]", label)
        for key in before:
            if not (key.startswith("CMC@") or key == "mAP"):
                continue
            b = before[key]
            a = after.get(key, b)
            d = after.get(f"delta_{key}", a - b)
            logger.info("    %s: %.4f → %.4f  (Δ %+.4f)", key, b, a, d)

    baseline   = metrics.get("baseline_all_queries",       {})
    after_all  = metrics.get("after_reranking_all_queries", {})
    sub_before = metrics.get("analyzed_subset_before",     {})
    sub_after  = metrics.get("analyzed_subset_after",      {})

    _fmt_tier("ALL QUERIES",      baseline,   after_all)
    if sub_before:
        _fmt_tier("ANALYZED SUBSET", sub_before, sub_after)

    subset_avg_before = metrics.get("subset_average_baseline", {})
    subset_avg_after  = metrics.get("subset_average_after_reranking", {})
    if subset_avg_before:
        _fmt_tier(
            f"PER-SUBSET AVERAGE ({metrics.get('n_subsets', '?')} subsets, equal weight)",
            subset_avg_before, subset_avg_after,
        )

    breakdown = metrics.get("rank_change_breakdown", {})
    if breakdown:
        logger.info("")
        logger.info("  [RANK CHANGE (processed subset)]")
        logger.info("    Improved : %d", breakdown.get("n_improved",  0))
        logger.info("    Degraded : %d", breakdown.get("n_degraded",  0))
        logger.info("    Unchanged: %d", breakdown.get("n_unchanged", 0))

    token_usage = metrics.get("token_usage", {})
    if token_usage and token_usage.get("num_api_calls", 0) > 0:
        logger.info("")
        logger.info("TOKEN USAGE")
        logger.info("-" * sep_width)
        logger.info("  API calls              : %d", token_usage["num_api_calls"])
        logger.info("  Total prompt tokens    : %d", token_usage["total_prompt_tokens"])
        logger.info("  Total completion tokens: %d", token_usage["total_completion_tokens"])
        logger.info("  Total tokens           : %d", token_usage["total_tokens"])
        logger.info("  Avg prompt tokens      : %d", token_usage["avg_prompt_tokens"])
        logger.info("  Avg completion tokens  : %d", token_usage["avg_completion_tokens"])
        logger.info("  Avg total tokens       : %d", token_usage["avg_total_tokens"])

    if "elapsed_time" in metrics:
        logger.info("")
        logger.info("  Elapsed time: %s", metrics["elapsed_time"])

    logger.info(sep)

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info("  Metrics saved to : %s", metrics_path)
