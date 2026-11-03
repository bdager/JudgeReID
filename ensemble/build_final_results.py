#!/usr/bin/env python3
"""
Assemble the final results for the agreement + reranked-disagreement pipeline
and report CMC/mAP metrics (via metrics/calc_metrics.py) per subset, combined,
and against a single-model baseline.

Pipeline recap:
    ensemble/top1_agreement.py --output agree.csv     agreement subset: full rank
                                                       already copied from one
                                                       "primary" model (--source-csv,
                                                       defaults to the first --results
                                                       file passed to top1_agreement.py)
    ensemble/merge_topk.py + eval_vlm_rerank.py       disagreement subset: produces
                                                       reranked_gallery_ids per query,
                                                       drawn from a merged multi-model
                                                       candidate pool (or a single
                                                       model's own top-k)

This script:
  0. Checks the two subsets exactly partition the baseline queries, and aborts
     otherwise -- metrics are means over whatever reaches combined.csv, so a
     dropped query silently inflates them instead of scoring 0 (see
     check_query_coverage).
  1. Takes the agreement-subset CSV as-is (already a full rank).
  2. For the disagreement subset, extends each query's reranked_gallery_ids
     (optionally truncated to --head-size first -- 'sort' mode returns a full
     permutation of every candidate given to it, not a fixed shortlist, so
     without this the "head" is n_candidates, not a clean top-N) with a
     duplicate-free tail drawn from whichever model contributed the most items to
     that query's final reranked top-k ("majority contributor"). The exact image
     at a given candidate position is resolved from --candidates-csv (the file
     actually passed as --results_csv to eval_vlm_rerank.py),
     and provenance is then decided by membership in each model's own top-k
     (k = --merge-k) rather than by the item's position in the pool: merge_topk.py
     keeps only the first copy of an image several models picked and renumbers
     ranks contiguously, so position alone always names the earliest model and
     silently biases the vote towards it. --shared-credit decides who gets an
     image that genuinely sat in more than one model's top-k.
  3. Concatenates both subsets into a combined set.
  4. Runs metrics/calc_metrics.py on: a single-model baseline (unreranked, all
     queries), the agreement subset, the no-agreement subset, and the combined set.

The paper's setting (docs/reproduce.md step 5):
    python ensemble/build_final_results.py \
        --results differ/long_term_results.csv sapiensid/long_term_results.csv \
        --names differ sapiensid \
        --agree-csv gate/long_term/agreed.csv \
        --candidates-csv gate/long_term/union_k8.csv \
        --rerank-csv rerank/qwen3.6-27b/long_term/t16/results.csv \
        --merge-k 8 --head-size 8 \
        --shared-credit all --shared-credit-tiebreak sapiensid --fallback-model sapiensid \
        --topk 1 3 5 10 20 --per-subset \
        --output-dir rerank/qwen3.6-27b/long_term/t16/final

The tail-rule ablation changes --tail-model, --shared-credit and
--shared-credit-tiebreak.
"""

import argparse
import json
import logging
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from utils.logging_config import setup_logging

CALC_METRICS = str(REPO_ROOT / "metrics" / "calc_metrics.py")
LOGGING_CONFIG = str(REPO_ROOT / "configs" / "logging_config.yaml")

OWN_RANK = "own-rank"
ALL_SOURCES = "all"

logger = logging.getLogger(__name__)


def normalize_path(p: str) -> str:
    """Extract the relative suffix starting from reid/... to align paths across models."""
    match = re.search(r"(reid/[^/]+/.+)$", p)
    return match.group(1) if match else p


def load_full_rank(path: str) -> dict:
    """query_key -> [(gallery_id, match_key), ...] sorted by rank."""
    df = pd.read_csv(path)
    df["query_key"] = df["query_path"].apply(normalize_path)
    df["match_key"] = df["gallery_path"].apply(normalize_path)
    df = df.sort_values(["query_key", "rank"])
    return {
        qk: list(zip(g["gallery_id"].astype(int), g["match_key"]))
        for qk, g in df.groupby("query_key")
    }


def assemble_no_agreement_subset(
    rerank_df: pd.DataFrame,
    full_rank: dict,
    candidates: dict,
    model_names: list,
    merge_k_arg: int | None,
    head_size: int | None = None,
    forced_tail_model: str | None = None,
    shared_credit: str | None = None,
    shared_credit_tiebreak: str | None = None,
):
    """Build tail-corrected full-rank rows for the no-agreement subset.

    forced_tail_model: pin the tail to one
    model's own original ranking for every query, instead of the per-query
    majority contributor. Isolates "whose rank order beyond the head do we
    trust" from "who won the merge", which the majority vote otherwise
    entangles.

    shared_credit: who gets counted as the contributor of a candidate that sat
    in more than one model's own top-k. merge_topk.py keeps only the first
    copy of such an image and renumbers the pool contiguously, so the item's
    position credits the earliest --names model by construction -- which biases
    the majority vote towards whichever model happens to be listed first.
    Either OWN_RANK (credit whichever model placed it higher in its own list)
    or a model name (always credit that model when it is one of the sources).
    Defaults to model_names[0], which reproduces the old positional behaviour.

    shared_credit_tiebreak: who wins a tie, at either of the two places one can
    arise -- under OWN_RANK, an image both models put at the *same* own rank (a
    quarter of shared items); and, for any policy, a query where the models end
    up with equal c_m (unreachable when each head item credits exactly one model
    and k is odd, but ~10% of gated queries under ALL_SOURCES). Left to --names
    order both quietly reinstate the first-model bias. Defaults to
    model_names[0], i.e. --names order.

    head_size: if given, only the first `head_size` items of the judge's
    reranked_gallery_ids are treated as "the head" -- the rest are dropped
    from the head and become ordinary tail territory (subject to the same
    dedup/refill as always). This matters because `sort` mode returns a full
    permutation of every candidate it was given (e.g. 16 items at P=16),
    not a fixed shortlist -- so without head_size, "the
    head" is whatever n_candidates was for that query, not a clean top-N.
    Everything below already operates on k_final generically, so setting
    head_size just truncates reranked_ids/reranked_ranks up front and the
    rest of the function is unchanged.

    Returns (rows, provenance_rows, n_gid_mismatch, n_missing_candidates).
    """
    rows = []
    provenance_rows = []
    n_gid_mismatch = 0
    n_missing_candidates = 0
    shared_credit = shared_credit or model_names[0]
    shared_credit_tiebreak = shared_credit_tiebreak or model_names[0]

    for _, row in rerank_df.iterrows():
        qpath = str(row["query_path"])
        qkey = normalize_path(qpath)
        query_id = int(row["query_id"])
        reranked_ids = json.loads(row["reranked_gallery_ids"])
        reranked_ranks = json.loads(row["reranked_ranks"])
        n_candidates = (
            int(row["n_candidates"])
            if "n_candidates" in row
            else len(reranked_ids)
        )
        if head_size is not None:
            reranked_ids = reranked_ids[:head_size]
            reranked_ranks = reranked_ranks[:head_size]
        k_final = len(reranked_ids)
        merge_k = merge_k_arg or max(1, n_candidates // len(model_names))

        cand_list = candidates.get(qkey)
        if cand_list is None:
            n_missing_candidates += 1
            cand_list = []

        # Each model's own top-merge_k for this query, as match_key -> own rank.
        # This is what merge_topk.py drew the candidate pool from, so membership
        # here -- not the item's position in the pool -- is the real provenance.
        own_topk_rank = {
            m: {
                mkey: r
                for r, (_gid, mkey) in enumerate(
                    full_rank.get(m, {}).get(qkey, [])[:merge_k]
                )
            }
            for m in model_names
        }

        # Resolve (gallery_id, match_key, source_model) for each item currently
        # in the new top-k.
        resolved = []
        contribution = Counter()
        n_shared = 0
        n_rank_ties = 0
        n_unresolved = 0
        for gid, rrank in zip(reranked_ids, reranked_ranks):
            if 1 <= rrank <= len(cand_list):
                cand_gid, cand_mkey = cand_list[rrank - 1]
                if cand_gid != gid:
                    n_gid_mismatch += 1
            else:
                cand_gid, cand_mkey = gid, None

            sources = (
                [m for m in model_names if cand_mkey in own_topk_rank[m]]
                if cand_mkey is not None
                else []
            )
            if len(sources) > 1:
                n_shared += 1
                if shared_credit == ALL_SOURCES:
                    # Eq. (2): c_m counts every head item with m in sigma(h_i),
                    # so a shared candidate credits each model that retrieved it.
                    source_model = sources[0]
                elif shared_credit == OWN_RANK:
                    best = min(own_topk_rank[m][cand_mkey] for m in sources)
                    tied = [
                        m for m in sources if own_topk_rank[m][cand_mkey] == best
                    ]
                    if len(tied) > 1:
                        n_rank_ties += 1
                        source_model = (
                            shared_credit_tiebreak
                            if shared_credit_tiebreak in tied
                            else tied[0]
                        )
                    else:
                        source_model = tied[0]
                elif shared_credit in sources:
                    source_model = shared_credit
                else:
                    source_model = sources[0]
            elif sources:
                source_model = sources[0]
            else:
                # Image unresolvable against the models' own top-k (candidate row
                # missing, or a --merge-k that doesn't match how the pool was
                # built). Fall back to the positional block -- correct for two
                # models, since dedup only shortens the last block.
                n_unresolved += 1
                block = min((rrank - 1) // merge_k, len(model_names) - 1)
                source_model = model_names[block]
            resolved.append((gid, cand_mkey, source_model))
            for m in sources if shared_credit == ALL_SOURCES and sources else (
                source_model,
            ):
                contribution[m] += 1

        top_c = max(contribution.get(m, 0) for m in model_names)
        vote_tie = sum(contribution.get(m, 0) == top_c for m in model_names) > 1
        if forced_tail_model is not None:
            tail_model = forced_tail_model
        else:
            # Majority vote, ties to shared_credit_tiebreak (then --names order).
            # Ties are unreachable when each head item credits exactly one model
            # and k is odd, but ALL_SOURCES lets both counters reach k, so this
            # decides ~10% of gated queries there.
            tail_model = max(
                model_names,
                key=lambda m: (
                    contribution.get(m, 0),
                    m == shared_credit_tiebreak,
                    -model_names.index(m),
                ),
            )

        reranked_match_keys = {
            mkey for _, mkey, _ in resolved if mkey is not None
        }

        tail_full = full_rank.get(tail_model, {}).get(qkey, [])
        tail_candidates = tail_full[k_final:]
        tail_own_topk = tail_full[:k_final]

        # Items from tail_model's own top-k that were displaced from the final
        # reranked head (by merging and/or VLM reranking) -- available to refill
        # tail slots vacated by dedup, in the tail model's original rank order.
        refill_pool = iter(
            item for item in tail_own_topk if item[1] not in reranked_match_keys
        )

        final_tail = []
        n_swaps = 0
        for gid, mkey in tail_candidates:
            if mkey in reranked_match_keys:
                n_swaps += 1
                replacement = next(refill_pool, None)
                if replacement is not None:
                    final_tail.append(replacement)
            else:
                final_tail.append((gid, mkey))

        tail_ids = [gid for gid, _ in final_tail]
        full_ids = reranked_ids + tail_ids
        for rank, gid in enumerate(full_ids, start=1):
            rows.append(
                {
                    "query_path": qpath,
                    "query_id": query_id,
                    "gallery_id": gid,
                    "rank": rank,
                }
            )

        provenance_rows.append(
            {
                "query_path": qpath,
                "query_id": query_id,
                "tail_model": tail_model,
                **{f"contrib_{m}": contribution.get(m, 0) for m in model_names},
                "n_shared_items": n_shared,
                "n_shared_rank_ties": n_rank_ties,
                "vote_tie": vote_tie,
                "n_unresolved_items": n_unresolved,
                "n_dedup_swaps": n_swaps,
            }
        )

    return rows, provenance_rows, n_gid_mismatch, n_missing_candidates


def check_query_coverage(
    baseline_csv: str,
    agree_df: pd.DataFrame,
    rerank_df: pd.DataFrame,
    eval_set: str,
) -> None:
    """Fail if the two subsets don't exactly partition the baseline queries.

    Metrics are plain means over whatever rows reach combined.csv, so a query
    missing from both subsets doesn't score 0 -- it leaves the denominator and
    inflates every number. The usual cause is a rerank run made without
    eval_vlm_rerank.py's --process_all, which sends only GT-in-top-K queries to
    the judge (exactly the hard queries whose absence inflates the metrics),
    or an interrupted resumable run. A query in *both* subsets is the opposite
    bug: it gets two full ranks concatenated into one list.
    """
    base = pd.read_csv(baseline_csv, usecols=["query_path", "query_id"])
    base["query_key"] = base["query_path"].apply(normalize_path)
    base = base.drop_duplicates("query_key")
    if eval_set == "closed":
        base = base[base["query_id"] >= 0]
    expected = set(base["query_key"])

    def keys(df):
        return set(df["query_path"].apply(normalize_path)) if len(df) else set()

    agree_keys, rerank_keys = keys(agree_df), keys(rerank_df)
    logger.info(
        "Query coverage (%s-set): %d baseline | %d agreement | %d reranked",
        eval_set,
        len(expected),
        len(agree_keys),
        len(rerank_keys),
    )

    overlap = agree_keys & rerank_keys
    if overlap:
        raise SystemExit(
            f"ERROR: {len(overlap)} queries appear in both --agree-csv and "
            f"--rerank-csv (e.g. {sorted(overlap)[0]}); they would be counted "
            "twice in combined.csv. Check both files come from the same "
            "top1_agreement.py split."
        )

    missing = expected - (agree_keys | rerank_keys)
    if missing:
        raise SystemExit(
            f"ERROR: {len(missing)}/{len(expected)} baseline queries are in "
            f"neither --agree-csv nor --rerank-csv (e.g. {sorted(missing)[0]}). "
            "They would be dropped from combined.csv, so every metric would be "
            "computed over a smaller, easier query set. Most likely the rerank "
            "run omitted --process_all (eval_vlm_rerank.py then processes only "
            "queries whose GT is already in the top-K) or was interrupted."
        )


def run_calc_metrics(
    results_csv: Path,
    out_json: Path,
    topk: list,
    eval_set: str,
    per_subset: bool = False,
) -> dict:
    cmd = [
        sys.executable,
        CALC_METRICS,
        "--results",
        str(results_csv),
        "--topk",
        *[str(k) for k in topk],
        "--eval-set",
        eval_set,
        "--output",
        str(out_json),
    ]
    if per_subset:
        cmd.append("--per-subset")
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    return json.loads(out_json.read_text())


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results",
        nargs="+",
        required=True,
        help="Per-model original full-rank CSVs, in the order used to build the "
        "rerank candidate pool.",
    )
    parser.add_argument(
        "--names",
        nargs="+",
        default=None,
        help="Model names matching --results order. Auto-derived if omitted.",
    )
    parser.add_argument(
        "--agree-csv",
        required=True,
        help="Output of top1_agreement.py --output (agreement-subset full rank, "
        "already sourced from the desired primary model).",
    )
    parser.add_argument(
        "--candidates-csv",
        required=True,
        help="The exact CSV passed as --results_csv to eval_vlm_rerank.py "
        "for the no-agreement subset (the merged candidate pool). Used to "
        "resolve the exact image at each reranked_ranks position.",
    )
    parser.add_argument(
        "--rerank-csv",
        required=True,
        help="results.csv produced by eval_vlm_rerank.py.",
    )
    parser.add_argument(
        "--merge-k",
        type=int,
        default=None,
        help="Per-model candidate-block size used when building the candidate "
        "pool (block i = reranked_ranks in [i*k+1, (i+1)*k] came from "
        "--names[i]). Defaults to n_candidates // len(--names); pass explicitly "
        "when known.",
    )
    parser.add_argument("--topk", nargs="+", type=int, default=[1, 5, 10])
    parser.add_argument(
        "--eval-set",
        dest="eval_set",
        choices=["open", "closed", "both"],
        default="closed",
    )
    parser.add_argument(
        "--per-subset",
        dest="per_subset",
        action="store_true",
        help="Also report CMC/mAP grouped by CHIRLA subset (test_1, test_2, "
        "...) plus their simple average for every metrics_*.json in the "
        "report -- matches metrics/evaluate_reid.py's --per-subset "
        "convention, so these final-results numbers are directly comparable "
        "to the rest of the paper's per-subset tables.",
    )
    parser.add_argument(
        "--head-size",
        type=int,
        default=None,
        help="Treat only the first N items of the judge's reranked_gallery_ids "
        "as the head; the rest fall through to tail territory. Needed because "
        "'sort' mode returns a full permutation of every candidate it was given "
        "(e.g. 16 items at P=16), not a fixed shortlist -- without "
        "this, CMC@k for k below n_candidates isn't a fair top-N comparison. "
        "Omit to use the full reranked_gallery_ids length.",
    )
    parser.add_argument(
        "--tail-model",
        dest="tail_model",
        default=None,
        help="Pin the tail of every no-agreement query to this model's own "
        "original ranking (must be one of --names), instead of the per-query "
        "majority contributor (tail-rule ablation).",
    )
    parser.add_argument(
        "--shared-credit",
        dest="shared_credit",
        default=None,
        help="Who is counted as the contributor of a candidate that was in more "
        "than one model's own top-k. merge_topk.py keeps one copy of such an "
        f"image, so its pool position always names the earliest model. '{ALL_SOURCES}' "
        "credits every model that retrieved it, which is what the paper's c_m "
        f"definition states; '{OWN_RANK}' credits whichever model placed it higher "
        "in its own ranking (exact ties go to --shared-credit-tiebreak); a model "
        "name from --names always credits that model. Only affects the per-query "
        "majority vote for the tail (and so is inert under --tail-model). Defaults "
        "to --names[0].",
    )
    parser.add_argument(
        "--shared-credit-tiebreak",
        dest="shared_credit_tiebreak",
        default=None,
        help="Which model wins a tie. Two can arise: with --shared-credit "
        f"{OWN_RANK}, an image both models placed at the same own rank (about a "
        "quarter of shared items); and, for any policy, a query where the models "
        f"end up with equal c_m (~10%% of gated queries under {ALL_SOURCES}, "
        "unreachable when each head item credits one model and the head size is "
        "odd). Must be one of --names. Defaults to --names[0] (--names order).",
    )
    parser.add_argument(
        "--fallback-model",
        dest="fallback_model",
        default=None,
        help="Queries whose rerank output was never valid (parse_status "
        "'fallback') get this model's full ranking instead of being assembled "
        "from the pool order. The pool order puts --results[0]'s block first, "
        "so without this the fallback depends on the pool ordering rho; the "
        "paper's rule is to fall back to m*. Must be one of --names. Omit to "
        "use the pool order.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--log-level", dest="log_level", default="INFO")
    args = parser.parse_args()

    setup_logging(config_path=LOGGING_CONFIG, log_level=args.log_level)

    if args.names:
        assert len(args.names) == len(
            args.results
        ), "Number of names must match number of results."
        model_names = args.names
    else:
        model_names = [Path(p).stem for p in args.results]
    if len(set(model_names)) != len(model_names):
        parser.error(
            f"Model names collide: {model_names}. Results files that share a "
            "filename (e.g. every model's own results.csv) auto-derive to the "
            "same name, which silently drops all but the last model's data. "
            "Pass --names explicitly (e.g. --names differ sapiensid)."
        )

    if args.tail_model is not None and args.tail_model not in model_names:
        parser.error(
            f"--tail-model {args.tail_model!r} is not one of --names ({model_names})."
        )

    shared_credit = args.shared_credit or model_names[0]
    if shared_credit not in (OWN_RANK, ALL_SOURCES) and shared_credit not in model_names:
        parser.error(
            f"--shared-credit {shared_credit!r} must be {OWN_RANK!r}, "
            f"{ALL_SOURCES!r}, or one of --names ({model_names})."
        )
    shared_credit_tiebreak = args.shared_credit_tiebreak or model_names[0]
    if shared_credit_tiebreak not in model_names:
        parser.error(
            f"--shared-credit-tiebreak {shared_credit_tiebreak!r} is not one of "
            f"--names ({model_names})."
        )

    if args.fallback_model is not None and args.fallback_model not in model_names:
        parser.error(
            f"--fallback-model {args.fallback_model!r} is not one of --names ({model_names})."
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Models (priority order): %s", model_names)
    logger.info(
        "Tail source: %s",
        args.tail_model or "per-query majority contributor",
    )
    if args.tail_model is None:
        logger.info(
            "Shared-candidate credit: %s (ties -> %s)",
            shared_credit,
            shared_credit_tiebreak,
        )

    agree_df = pd.read_csv(args.agree_csv)
    rerank_df = pd.read_csv(args.rerank_csv)
    check_query_coverage(args.results[0], agree_df, rerank_df, args.eval_set)

    full_rank = {
        name: load_full_rank(path)
        for name, path in zip(model_names, args.results)
    }
    candidates = load_full_rank(args.candidates_csv)

    # Unrecovered outputs carry no judge decision, so they leave the rerank
    # path and are scored like ungated queries: m*'s own full ranking.
    if args.fallback_model is not None and "parse_status" in rerank_df.columns:
        is_fb = rerank_df["parse_status"].eq("fallback")
        fb_rank = full_rank[args.fallback_model]
        fb_rows = []
        for _, row in rerank_df[is_fb].iterrows():
            ranked = fb_rank[normalize_path(str(row["query_path"]))]
            fb_rows += [
                {"query_path": row["query_path"], "query_id": int(row["query_id"]),
                 "gallery_id": gid, "rank": r}
                for r, (gid, _mkey) in enumerate(ranked, start=1)
            ]
        rerank_df = rerank_df[~is_fb]
        fallback_df = pd.DataFrame(fb_rows, columns=["query_path", "query_id", "gallery_id", "rank"])
        logger.info(
            "Fallback outputs:     %d queries -> %s full ranking",
            int(is_fb.sum()),
            args.fallback_model,
        )
    else:
        fallback_df = pd.DataFrame(columns=["query_path", "query_id", "gallery_id", "rank"])

    # ------------------------------------------------------------------
    # Agreement subset: copy as-is (already full rank, from top1_agreement.py)
    # ------------------------------------------------------------------
    agree_out = out_dir / "agreement_subset.csv"
    agree_df.to_csv(agree_out, index=False)
    n_agree_queries = agree_df["query_path"].apply(normalize_path).nunique()
    logger.info(
        "Agreement subset:    %d queries, %d rows -> %s",
        n_agree_queries,
        len(agree_df),
        agree_out,
    )

    # ------------------------------------------------------------------
    # No-agreement subset: tail-correct each query's reranked_gallery_ids
    # ------------------------------------------------------------------
    no_agree_rows, provenance_rows, n_mismatch, n_missing = assemble_no_agreement_subset(
        rerank_df,
        full_rank,
        candidates,
        model_names,
        args.merge_k,
        args.head_size,
        args.tail_model,
        shared_credit,
        shared_credit_tiebreak,
    )
    # Fallback queries are still gated, so they count in the no-agreement subset.
    no_agree_df = pd.concat(
        [pd.DataFrame(no_agree_rows), fallback_df], ignore_index=True
    )
    no_agree_out = out_dir / "no_agreement_subset.csv"
    no_agree_df.to_csv(no_agree_out, index=False)
    n_no_agree_queries = (
        no_agree_df["query_path"].apply(normalize_path).nunique()
        if len(no_agree_df)
        else 0
    )
    logger.info(
        "No-agreement subset: %d queries, %d rows -> %s",
        n_no_agree_queries,
        len(no_agree_df),
        no_agree_out,
    )
    if n_mismatch:
        logger.warning(
            "%d resolved gallery_id mismatches between --rerank-csv and "
            "--candidates-csv -- check --candidates-csv is the exact file used "
            "to produce --rerank-csv.",
            n_mismatch,
        )
    if n_missing:
        logger.warning(
            "%d queries in --rerank-csv were not found in --candidates-csv; "
            "their tail dedup was skipped for those unresolved items.",
            n_missing,
        )

    provenance_out = out_dir / "provenance_report.csv"
    provenance_df = pd.DataFrame(provenance_rows)
    provenance_df.to_csv(provenance_out, index=False)
    logger.info("Provenance report    -> %s", provenance_out)
    if len(provenance_df):
        logger.info(
            "Shared candidates: %d items over %d/%d queries (credited to %s); "
            "%d own-rank ties, %d vote ties -> %s, %d unresolved -> positional",
            int(provenance_df["n_shared_items"].sum()),
            int((provenance_df["n_shared_items"] > 0).sum()),
            len(provenance_df),
            shared_credit,
            int(provenance_df["n_shared_rank_ties"].sum()),
            int(provenance_df["vote_tie"].sum()),
            shared_credit_tiebreak,
            int(provenance_df["n_unresolved_items"].sum()),
        )
        logger.info(
            "Tail model distribution: %s",
            provenance_df["tail_model"].value_counts().to_dict(),
        )

    # ------------------------------------------------------------------
    # Combined
    # ------------------------------------------------------------------
    combined_out = out_dir / "combined.csv"
    pd.concat(
        [
            agree_df[["query_path", "query_id", "gallery_id", "rank"]],
            no_agree_df,
        ],
        ignore_index=True,
    ).to_csv(combined_out, index=False)
    logger.info("Combined:             -> %s", combined_out)

    # ------------------------------------------------------------------
    # Metrics: baseline / agreement / no-agreement / combined
    # ------------------------------------------------------------------
    logger.info("Computing metrics via metrics/calc_metrics.py ...")
    report = {}
    report["baseline"] = run_calc_metrics(
        Path(args.results[0]),
        out_dir / "metrics_baseline.json",
        args.topk,
        args.eval_set,
        per_subset=args.per_subset,
    )
    report["agreement_subset"] = run_calc_metrics(
        agree_out,
        out_dir / "metrics_agreement.json",
        args.topk,
        args.eval_set,
        per_subset=args.per_subset,
    )
    if len(no_agree_df):
        report["no_agreement_subset"] = run_calc_metrics(
            no_agree_out,
            out_dir / "metrics_no_agreement.json",
            args.topk,
            args.eval_set,
            per_subset=args.per_subset,
        )
    report["combined"] = run_calc_metrics(
        combined_out,
        out_dir / "metrics_combined.json",
        args.topk,
        args.eval_set,
        per_subset=args.per_subset,
    )
    report_out = out_dir / "final_report.json"
    report_out.write_text(json.dumps(report, indent=2))
    logger.info("Final report -> %s", report_out)
    logger.info("\n%s", json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
