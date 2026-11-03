#!/usr/bin/env python3
"""
VLM judge: re-rank each query's candidate pool with a vision-language model.

Reads a long-format candidate CSV (one row per query and candidate, e.g. the
union written by ensemble/merge_topk.py), shows the VLM the Query crop followed
by its top-K candidates as separate images, and asks for the candidates sorted
from most to least likely to be the Query person. Answers that are not a valid
permutation get a correction prompt (configs/config.yaml ``vlm.max_retries``);
if that fails too, the query keeps its original order (parse_status
'fallback').

By default only closed-set queries (query_id >= 0) whose ground truth is in the
top-K are sent, since the VLM cannot recover a match that is not in the pool.
--process_all sends every closed-set query, as a deployed system would.

Outputs in --output_dir: results.csv (one row per query, used by
ensemble/build_final_results.py), results.txt (full responses), unknowns.txt
(unparsed responses), metrics.json, run_stats.json and the run's command,
prompts and configuration. Re-running the same command resumes the run.

Usage:
    python eval_vlm_rerank.py \\
        --results_csv  union_k8.csv \\
        --topk         16 \\
        --vlm          qwen3.6-27b \\
        --data_root    $CHIRLA_ROOT/benchmark/ \\
        --label_images --process_all --retry_fallbacks \\
        --output_dir   rerank/long_term/t16

Input results_csv format (metrics/evaluate_reid.py --out-csv):
    query_path, query_id, gallery_path, gallery_id, rank, score
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import os
from typing import Dict, List

import pandas as pd
from dotenv import load_dotenv
from omegaconf import OmegaConf

from metrics.calc_metrics import rerank_metrics
from metrics.run_stats import RunStats, install_sigterm_handler, parse_stats_from_csv
from metrics.save_metrics import log_and_save_rerank_metrics
from reasoning.utils.common import save_reproducibility_artifacts
from reasoning.utils.concurrency import run_concurrent
from reasoning.utils.rerank_parsing import (
    CSV_FIELDS,
    SEP,
    build_example_line,
    build_result_row,
    build_vlm_placeholders,
    fill_prompt_placeholders,
    load_checkpoint,
    resolve_path,
    run_attempt_with_retries_async,
)
from reasoning.vlm_config import resolve_vlm_params
from reasoning.vlm_judge import VLMJudge
from utils.logging_config import setup_logging

load_dotenv()

logger = logging.getLogger(__name__)

CORRECTION_PROMPT = "reasoning/prompts/correction_rerank.txt"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VLM judge re-ranking of Re-ID candidate pools")
    p.add_argument(
        "--results_csv",
        required=True,
        help="Long-format candidate CSV "
        "(columns: query_path, query_id, gallery_path, gallery_id, rank, score)",
    )
    p.add_argument("--output_dir", required=True, help="Directory for all outputs")
    p.add_argument(
        "--query_csv",
        default=None,
        help="Optional CHIRLA query CSV; if given, only its queries are processed",
    )
    p.add_argument(
        "--topk",
        type=int,
        default=16,
        help="Pool size P: the first P candidates of each query are shown to the VLM",
    )
    p.add_argument(
        "--vlm",
        default="qwen3.6-27b",
        help="VLM config name (configs/vlm/{name}.yaml)",
    )
    p.add_argument(
        "--data_root",
        default=None,
        help="Prepended to relative image paths in the CSVs",
    )
    p.add_argument(
        "--sys_prompt",
        default="reasoning/prompts/sys_rerank.txt",
        help="System prompt (variants used in the paper's ablations are in "
        "reasoning/prompts/ablations/)",
    )
    p.add_argument(
        "--user_prompt",
        default="reasoning/prompts/user_rerank.txt",
        help="User prompt",
    )
    p.add_argument(
        "--user_prompt_position",
        choices=["before", "after"],
        default="before",
        help="Place the user instructions before or after the images",
    )
    p.add_argument(
        "--label_images",
        action="store_true",
        help="Burn a 'Query' / candidate-number banner into each image",
    )
    p.add_argument(
        "--topk_metrics",
        nargs="+",
        type=int,
        default=[1, 3, 5, 10, 20],
        help="K values reported as CMC@K in metrics.json",
    )
    p.add_argument(
        "--max_queries",
        type=int,
        default=None,
        help="Process at most this many queries (for a quick test)",
    )
    p.add_argument(
        "--process_all",
        action="store_true",
        help="Send every query to the VLM, including those whose ground truth is "
        "not in the top-K (default: skip them). metrics.json reports both the "
        "whole set and the GT-in-top-K subset.",
    )
    p.add_argument(
        "--eval_set",
        choices=["closed", "open"],
        default="closed",
        help="'closed' (default): only queries with query_id >= 0. Distractor "
        "queries (query_id < 0) have no correct gallery match, so judging them "
        "is wasted calls. 'open': keep every query.",
    )
    p.add_argument(
        "--retry_fallbacks",
        action="store_true",
        help="When resuming, re-process queries whose answer could not be parsed "
        "and that have retries left",
    )
    p.add_argument(
        "--per_subset",
        action="store_true",
        help="Also report CMC/mAP per CHIRLA subset (test_1, test_2, ...) and "
        "their simple average, as metrics/evaluate_reid.py --per-subset does",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Max concurrent API requests. Overrides configs/config.yaml "
        "vlm.concurrency.",
    )
    p.add_argument("--log_level", default="INFO")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = _parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(log_level=args.log_level)
    install_sigterm_handler()
    topk = args.topk

    # ── Prompts ────────────────────────────────────────────────────────────
    vlm_placeholders = build_vlm_placeholders(args.user_prompt_position, args.label_images)
    with open(args.sys_prompt, "r", encoding="utf-8") as f:
        sys_text = fill_prompt_placeholders(f.read().strip(), vlm_placeholders)
    with open(args.user_prompt, "r", encoding="utf-8") as f:
        user_text = fill_prompt_placeholders(f.read().strip(), vlm_placeholders)
    with open(CORRECTION_PROMPT, "r", encoding="utf-8") as f:
        correction_text = f.read().strip()

    # ── VLM config ─────────────────────────────────────────────────────────
    cfg = OmegaConf.load("configs/config.yaml")
    cfg.vlm.name = args.vlm
    vlm_params = dict(resolve_vlm_params(cfg))
    model = vlm_params.pop("model", None)
    base_url = vlm_params.pop("base_url", None)
    max_retries = vlm_params.pop("max_retries", 2)
    max_retries_delay = vlm_params.pop("max_retries_delay", 2)
    config_concurrency = vlm_params.pop("concurrency", 8)
    concurrency = args.concurrency if args.concurrency is not None else config_concurrency
    label_banner_h = vlm_params.pop("label_banner_h", 32)
    if not model:
        raise ValueError(f"No model in configs/vlm/{args.vlm}.yaml")

    # ── Reproducibility artefacts ──────────────────────────────────────────
    save_reproducibility_artifacts(
        args.output_dir,
        args,
        prompts={
            "system": sys_text,
            "user": user_text,
            "correction": correction_text.replace("{N}", str(topk)).replace(
                "{EXAMPLE_LINE}", build_example_line(topk)
            ),
        },
        extra={"model": model, "base_url": base_url, "vlm_params": vlm_params},
    )

    # ── Load the candidate pools ───────────────────────────────────────────
    logger.info("Loading results CSV: %s", args.results_csv)
    results_df = pd.read_csv(args.results_csv)
    required = {"query_path", "query_id", "gallery_path", "gallery_id", "rank"}
    if not required.issubset(results_df.columns):
        raise ValueError(f"results_csv missing columns: {required - set(results_df.columns)}")

    results_df["query_path"] = results_df["query_path"].apply(
        lambda p: resolve_path(str(p), args.data_root)
    )
    results_df["gallery_path"] = results_df["gallery_path"].apply(
        lambda p: resolve_path(str(p), args.data_root)
    )

    query_info: Dict[str, dict] = {}
    for qpath, grp in results_df.groupby("query_path"):
        grp = grp.sort_values("rank")
        all_candidates = [
            {
                "gallery_path": str(row["gallery_path"]),
                "gallery_id": int(row["gallery_id"]),
                "rank": int(row["rank"]),
            }
            for _, row in grp.iterrows()
        ]
        query_info[str(qpath)] = {
            "query_id": int(grp["query_id"].iloc[0]),
            "all_candidates": all_candidates,
            "candidates": all_candidates[:topk],
        }
    logger.info("Total queries in results_csv: %d", len(query_info))

    if args.query_csv:
        qdf = pd.read_csv(args.query_csv)
        valid = {resolve_path(str(p), args.data_root) for p in qdf["image_path"]}
        before = len(query_info)
        query_info = {k: v for k, v in query_info.items() if k in valid}
        logger.info("Filtered by query_csv: %d → %d queries", before, len(query_info))

    # Every query enters the metrics; only some are sent to the VLM
    all_queries_list: List[dict] = []
    for qpath, info in query_info.items():
        topk_ids = [c["gallery_id"] for c in info["candidates"]]
        all_queries_list.append(
            {
                "query_path": qpath,
                "query_id": info["query_id"],
                "original_ranked_ids": [c["gallery_id"] for c in info["all_candidates"]],
                "candidates": info["candidates"],
                "gt_in_topk": info["query_id"] in topk_ids,
                "reranked_ids": None,
            }
        )

    if args.eval_set == "closed":
        n_before = len(all_queries_list)
        all_queries_list = [q for q in all_queries_list if q["query_id"] >= 0]
        logger.info(
            "eval_set=closed: %d → %d queries (dropped %d with query_id < 0)",
            n_before, len(all_queries_list), n_before - len(all_queries_list),
        )

    n_total = len(all_queries_list)
    n_analyzable = sum(1 for q in all_queries_list if q["gt_in_topk"])
    logger.info(
        "Queries: %d total | %d with GT in top-%d | %d without",
        n_total, n_analyzable, topk, n_total - n_analyzable,
    )

    to_process = (
        all_queries_list if args.process_all
        else [q for q in all_queries_list if q["gt_in_topk"]]
    )
    if args.max_queries is not None:
        to_process = to_process[: args.max_queries]
        logger.info("Capped to %d queries (--max_queries)", len(to_process))

    # ── Resume ─────────────────────────────────────────────────────────────
    results_csv_path = os.path.join(args.output_dir, "results.csv")
    results_txt_path = os.path.join(args.output_dir, "results.txt")
    unknowns_path = os.path.join(args.output_dir, "unknowns.txt")

    qpath_to_entry: Dict[str, dict] = {q["query_path"]: q for q in all_queries_list}
    done_qpaths, restored_reranked = load_checkpoint(
        results_csv_path, max_retries, args.retry_fallbacks, results_txt_path, unknowns_path,
    )
    for qp, rids in restored_reranked.items():
        if qp in qpath_to_entry:
            qpath_to_entry[qp]["reranked_ids"] = rids

    todo = [q for q in to_process if q["query_path"] not in done_qpaths]
    logger.info("Queries remaining: %d", len(todo))

    judge = VLMJudge(
        model=model,
        base_url=base_url,
        temperature=vlm_params.pop("temperature", 0),
        max_tokens=vlm_params.pop("max_tokens", 1024),
        top_p=vlm_params.pop("top_p", 1.0),
        label_banner_h=label_banner_h,
        **vlm_params,
    )

    # A few labeled inputs, to check by eye what the VLM receives
    if args.label_images and to_process:
        labeled_dir = os.path.join(args.output_dir, "labeled_imgs")
        os.makedirs(labeled_dir, exist_ok=True)
        n_examples = min(5, len(to_process))
        for ex_idx in range(n_examples):
            example_q = to_process[ex_idx]
            judge.label_image(example_q["query_path"], "Query").save(
                os.path.join(labeled_dir, f"query_{ex_idx}.jpg")
            )
            for i, cand in enumerate(example_q["candidates"]):
                judge.label_image(cand["gallery_path"], str(i + 1)).save(
                    os.path.join(labeled_dir, f"candidate_{i + 1}_{ex_idx}.jpg")
                )
        logger.info("Saved %d labeled example set(s) to %s", n_examples, labeled_dir)

    # ── Output files (appended to when resuming) ───────────────────────────
    csv_file = open(results_csv_path, "a" if done_qpaths else "w", newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    if not done_qpaths:
        csv_writer.writeheader()
    txt_file = open(results_txt_path, "a", encoding="utf-8")
    unk_file = open(unknowns_path, "a", encoding="utf-8")

    # Time and tokens accumulate in run_stats.json; the rest comes from the CSV
    stats = RunStats(args.output_dir)
    semaphore = asyncio.Semaphore(concurrency)

    async def process_one(q: dict) -> dict:
        candidates = q["candidates"]
        n = len(candidates)
        query_user_text = user_text.replace("{N}", str(n)).replace(
            "{EXAMPLE_LINE}", build_example_line(n)
        )
        reranked_cands, parse_status, num_retries = list(candidates), "fallback", 0
        response, usages = "", []
        try:
            messages = judge.build_rerank_prompt(
                q["query_path"],
                [c["gallery_path"] for c in candidates],
                sys_text,
                query_user_text,
                user_prompt_position=args.user_prompt_position,
                label_images=args.label_images,
            )
            reranked_cands, parse_status, num_retries, response, usages = (
                await run_attempt_with_retries_async(
                    judge, messages, candidates, correction_text,
                    max_retries, max_retries_delay, semaphore,
                )
            )
        except Exception as exc:
            logger.error("Unexpected error for %s: %s", q["query_path"], exc)
        return {
            "qpath": q["query_path"],
            "query_id": q["query_id"],
            "orig_ids": [c["gallery_id"] for c in candidates],
            "reranked_ids": [c["gallery_id"] for c in reranked_cands],
            "reranked_ranks": [c["rank"] for c in reranked_cands],
            "n": n,
            "parse_status": parse_status,
            "num_retries": num_retries,
            "response": response,
            "usages": usages,
        }

    n_completed = 0

    def on_result(result: dict) -> None:
        nonlocal n_completed
        qpath, query_id = result["qpath"], result["query_id"]
        stats.add_usages(result["usages"])
        qpath_to_entry[qpath]["reranked_ids"] = result["reranked_ids"]

        row = build_result_row(
            qpath, query_id, result["orig_ids"], result["reranked_ids"],
            result["reranked_ranks"], result["n"], result["parse_status"],
            result["num_retries"],
        )
        if result["parse_status"] == "fallback":
            unk_file.write(f"query: {qpath}\nquery_id: {query_id}\nresponse:\n{result['response']}\n{SEP}\n")
            unk_file.flush()
        csv_writer.writerow(row)
        csv_file.flush()
        txt_file.write(
            f"query: {qpath}\nquery_id: {query_id}\n"
            f"original_gt_rank: {row['original_gt_rank']}  "
            f"reranked_gt_rank: {row['reranked_gt_rank']}\n"
            f"parse_status: {result['parse_status']}  num_retries: {result['num_retries']}\n"
            f"response:\n{result['response']}\n{SEP}\n"
        )
        txt_file.flush()

        # Saving after every query means a hard kill loses at most one answer
        n_completed += 1
        stats.save()

    # ── Main loop (an interrupt still reaches the metrics below) ──────────
    interrupted: BaseException | None = None
    try:
        asyncio.run(run_concurrent(todo, process_one, on_result, desc="Re-ranking queries"))
    except KeyboardInterrupt as exc:
        interrupted = exc
        logger.warning(
            "Interrupted (%s) after %d/%d queries — saving partial results",
            exc, n_completed, len(todo),
        )
    except Exception as exc:
        interrupted = exc
        logger.exception("Run failed after %d/%d queries — saving partial results", n_completed, len(todo))
    finally:
        csv_file.close()
        txt_file.close()
        unk_file.close()
        stats.save()

    # The judged top-K followed by the rest of the original ranking
    for q in all_queries_list:
        if q["reranked_ids"] is not None:
            q["reranked_ids"] = q["reranked_ids"] + q["original_ranked_ids"][topk:]

    # ── Metrics ────────────────────────────────────────────────────────────
    metrics = rerank_metrics(
        all_queries_list,
        topk_values=args.topk_metrics,
        elapsed_secs=stats.elapsed_secs,
        token_usage=stats.token_usage,
        parse_stats=parse_stats_from_csv(results_csv_path),
        per_subset=args.per_subset,
    )
    if interrupted is not None:
        metrics["interrupted"] = f"{type(interrupted).__name__}: {interrupted}"
    log_and_save_rerank_metrics(args.output_dir, metrics)

    if interrupted is not None:
        logger.warning("metrics.json covers a partial run — re-run the same command to resume")
        raise SystemExit(130 if isinstance(interrupted, KeyboardInterrupt) else 1)


if __name__ == "__main__":
    main()
