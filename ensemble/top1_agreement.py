#!/usr/bin/env python3
"""
Analyze top-1 agreement across multiple models.

For each query, checks whether all models agree on the same gallery_id at rank 1.
When they agree, reports whether the prediction is correct (gallery_id == query_id).

Usage:
    python ensemble/top1_agreement.py \
        --results path/to/model_a/results.csv path/to/model_b/results.csv ...
"""

import argparse
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import pandas as pd


def normalize_path(p: str) -> str:
    """Extract the relative suffix starting from reid/... to align paths across models."""
    match = re.search(r"(reid/[^/]+/.+)$", p)
    return match.group(1) if match else p


def extract_subset_from_path(path: str) -> str:
    """Mirror of metrics/evaluate_reid.py::extract_subset_from_path, duplicated
    here rather than imported so this script keeps working when run directly
    as `python ensemble/top1_agreement.py` (repo root isn't on sys.path then,
    only ensemble/ is)."""
    if "/test/" in path or "/train/" in path:
        parts = path.split("/")
        for i, part in enumerate(parts):
            if part in ["test", "train"] and i + 1 < len(parts):
                next_part = parts[i + 1]
                if next_part.startswith(("test_", "train_")):
                    return next_part
        return "unknown"
    else:
        filename = Path(path).name
        parts = filename.split("_")
        if len(parts) >= 2:
            return f"{parts[0]}_{parts[1]}"
        return "unknown"


def get_top1(csv_path: str, match_by: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    cols = ["query_path", "query_id", "gallery_id"]
    if match_by == "image":
        cols.append("gallery_path")
    top1 = df[df["rank"] == 1][cols].copy()
    top1["query_key"] = top1["query_path"].apply(normalize_path)
    if match_by == "image":
        top1["match_key"] = top1["gallery_path"].apply(normalize_path)
    else:
        top1["match_key"] = top1["gallery_id"]
    top1 = top1.sort_values("query_key").reset_index(drop=True)
    return top1


def get_query_candidates(csv_path: str) -> pd.DataFrame:
    """For each query: query_id, and (rank, match_key, is_correct) for every
    candidate gallery row, sorted by rank. match_key identifies the physical
    gallery image normalized gallery_path), used to de-duplicate candidates when
    combining models."""
    df = pd.read_csv(csv_path)
    df["query_key"] = df["query_path"].apply(normalize_path)
    df["match_key"] = df["gallery_path"].apply(normalize_path)
    df["is_correct"] = df["gallery_id"] == df["query_id"]
    df = df.sort_values(["query_key", "rank"])
    grouped = (
        df.groupby("query_key")
        .apply(
            lambda g: list(
                zip(
                    g["rank"].tolist(),
                    g["match_key"].tolist(),
                    g["is_correct"].tolist(),
                )
            ),
            include_groups=False,
        )
        .rename("candidates")
    )
    queries = df.drop_duplicates("query_key")[["query_key", "query_id"]]
    result = queries.merge(grouped, on="query_key", how="left")
    result["candidates"] = result["candidates"].apply(
        lambda c: c if isinstance(c, list) else []
    )
    return result


def first_hit_rank(candidates: list) -> float:
    """Rank of the first correct candidate, or NaN if none."""
    return min(
        (r for r, _, correct in candidates if correct), default=float("nan")
    )


def first_hit_rank_series(candidates: pd.Series) -> pd.Series:
    return candidates.apply(first_hit_rank)


def precision_at_k(candidates: pd.Series, k: int) -> tuple[float, int]:
    """Precision@k: mean, over queries, of (# correct candidates in top-k) /
    min(k, # candidates available) — distinct from CMC, which only asks if there was
    >= 1 hit within the top-k."""
    total = len(candidates)
    if total == 0:
        return 0.0, 0

    def per_query_precision(candidate_list: list) -> float:
        window = candidate_list[:k]
        if not window:
            return 0.0
        n_hit = sum(1 for _, _, correct in window if correct)
        return n_hit / len(window)

    mean_precision = float(candidates.apply(per_query_precision).mean())
    return mean_precision, total


def cmc_at_k_values(first_hit_ranks: pd.Series, k_values: list) -> list[float]:
    """CMC@k (%) for each requested k — cumulative fraction of queries with >= 1 correct
    match by rank k."""
    total = len(first_hit_ranks)
    if total == 0:
        return [0.0] * len(k_values)
    return [100 * float((first_hit_ranks <= k).sum()) / total for k in k_values]


def print_cmc_at_k_values(
    first_hit_ranks: pd.Series, k_values: list, indent: int = 30
) -> None:
    curve = cmc_at_k_values(first_hit_ranks, k_values)
    for k, pct in zip(k_values, curve):
        label = f"  → CMC@{k}"
        print(f"  {label:>{indent}}: {pct:.1f}%")


def per_subset_top1_metrics(
    top1_df: pd.DataFrame, candidates_df: pd.DataFrame, k_values: list
) -> list[dict]:
    """Group a single model's top1s/candidates rows by CHIRLA subset (test_N,
    parsed from query_key) and compute accuracy, precision@k, and CMC@k per
    subset -- equal-weight-per-subset convention, matching
    metrics/evaluate_reid.py's --per-subset."""
    top1_df = top1_df.assign(
        subset=top1_df["query_key"].apply(extract_subset_from_path)
    )
    candidates_df = candidates_df.assign(
        subset=candidates_df["query_key"].apply(extract_subset_from_path)
    )
    subsets = sorted(set(top1_df["subset"]) | set(candidates_df["subset"]))
    rows = []
    for subset in subsets:
        sub_top1 = top1_df[top1_df["subset"] == subset]
        sub_cand = candidates_df.loc[
            candidates_df["subset"] == subset, "candidates"
        ]
        n = len(sub_top1)
        if n == 0:
            continue
        correct = int((sub_top1["query_id"] == sub_top1["gallery_id"]).sum())
        row = {"subset": subset, "n_queries": n, "accuracy": correct / n}
        for k in k_values:
            precision, _ = precision_at_k(sub_cand, k)
            row[f"precision@{k}"] = precision
        cmc_curve = cmc_at_k_values(first_hit_rank_series(sub_cand), k_values)
        for k, pct in zip(k_values, cmc_curve):
            row[f"CMC@{k}"] = pct / 100
        rows.append(row)
    return rows


def print_per_subset_top1_metrics(
    rows: list[dict], k_values: list, indent: int = 30
) -> None:
    if not rows:
        return
    for row in rows:
        subset_label = f"    {row['subset']}"
        print(
            f"  {subset_label:>{indent}}: "
            f"{row['n_queries']} queries, accuracy={100*row['accuracy']:.1f}%"
        )
        for k in k_values:
            print(
                f"  {f'      precision@{k}':>{indent}}: {100*row[f'precision@{k}']:.1f}%"
                f"   CMC@{k}: {100*row[f'CMC@{k}']:.1f}%"
            )
    n = len(rows)
    avg_acc = sum(r["accuracy"] for r in rows) / n
    print(
        f"  {'    Simple Average':>{indent}}: accuracy={100*avg_acc:.1f}% "
        f"({n} subsets, equal weight)"
    )
    for k in k_values:
        avg_p = sum(r[f"precision@{k}"] for r in rows) / n
        avg_c = sum(r[f"CMC@{k}"] for r in rows) / n
        print(
            f"  {f'      precision@{k}':>{indent}}: {100*avg_p:.1f}%"
            f"   CMC@{k}: {100*avg_c:.1f}%"
        )


def print_per_subset_agreement(
    subset_stats: dict[str, dict[str, int]], indent: int = 30
) -> None:
    """Print per-subset agreement rate + correct-when-agreeing rate, plus
    their simple (equal-weight-per-subset) average."""
    if not subset_stats:
        return
    agree_rates = []
    correct_rates = []
    for subset in sorted(subset_stats):
        s = subset_stats[subset]
        n_total, n_agree_s = s["n_total"], s["n_agree"]
        agree_rate = n_agree_s / n_total if n_total else 0.0
        agree_rates.append(agree_rate)
        label = f"    {subset}"
        line = (
            f"  {label:>{indent}}: agree {n_agree_s}/{n_total} "
            f"({100*agree_rate:.1f}%)"
        )
        if n_agree_s > 0:
            correct_rate = s["n_agree_correct"] / n_agree_s
            correct_rates.append(correct_rate)
            line += f"   correct-when-agree {100*correct_rate:.1f}%"
        print(line)
    avg_agree = sum(agree_rates) / len(agree_rates)
    label = "    Simple Average"
    line = (
        f"  {label:>{indent}}: agree {100*avg_agree:.1f}% "
        f"({len(agree_rates)} subsets, equal weight)"
    )
    if correct_rates:
        avg_correct = sum(correct_rates) / len(correct_rates)
        line += f"   correct-when-agree {100*avg_correct:.1f}%"
    print(line)


def main():
    parser = argparse.ArgumentParser(
        description="Top-1 agreement analysis across models."
    )
    parser.add_argument(
        "--results",
        nargs="+",
        required=True,
        help="Paths to results CSV files (one per model).",
    )
    parser.add_argument(
        "--names",
        nargs="+",
        default=None,
        help="Model names (same order as --results). Auto-derived from paths if omitted.",
    )
    parser.add_argument(
        "--open-map-threshold",
        dest="open_map_threshold",
        type=float,
        default=None,
        help="When set, include queries with negative IDs (unknown identities). "
        "By default they are excluded (closed-set evaluation).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save a CSV with rows where all models agree on top-1.",
    )
    parser.add_argument(
        "--source-csv",
        dest="source_csv",
        default=None,
        help="Which results CSV to copy rows from for the output file. "
        "Must be one of the --results paths. If omitted, defaults to the first.",
    )
    parser.add_argument(
        "--match-by",
        dest="match_by",
        choices=["id", "image"],
        default="id",
        help="Basis for top-1 agreement: 'id' (default) agrees when models predict the same "
        "identity (gallery_id); 'image' agrees only when models pick the same gallery "
        "image (gallery_path).",
    )
    parser.add_argument(
        "--topk",
        dest="topk",
        nargs="+",
        type=int,
        default=[3],
        help="Top-k values to report accuracy for on the disagreement subset, e.g. --topk 3 6.",
    )
    parser.add_argument(
        "--per-subset",
        action="store_true",
        help="Also break down the top-1 agreement rate (+ correct/wrong "
        "when agreeing) and 'Individual model top-1 accuracy' (accuracy, "
        "precision@k, CMC@k) by CHIRLA subset (test_1, test_2, ...) plus "
        "their simple average -- equal-weight-per-subset convention, "
        "matching metrics/evaluate_reid.py's --per-subset.",
    )
    args = parser.parse_args()

    if args.names:
        assert len(args.names) == len(
            args.results
        ), "Number of names must match number of results."
        model_names = args.names
    else:
        model_names = ["/".join(Path(p).parts[-3:-1]) for p in args.results]

    top1s = {}
    candidates = {}
    for name, path in zip(model_names, args.results):
        top1s[name] = get_top1(path, args.match_by)
        candidates[name] = get_query_candidates(path)

    if args.open_map_threshold is None:
        for name in model_names:
            before = len(top1s[name])
            top1s[name] = top1s[name][top1s[name]["query_id"] >= 0].reset_index(
                drop=True
            )
            after = len(top1s[name])
            if before != after:
                print(
                    f"  {name}: filtered {before - after} unknown (negative ID) queries"
                )
            candidates[name] = candidates[name][
                candidates[name]["query_id"] >= 0
            ].reset_index(drop=True)
        print()

    candidate_lookup = {
        name: dict(
            zip(candidates[name]["query_key"], candidates[name]["candidates"])
        )
        for name in model_names
    }

    ref_name = model_names[0]
    query_keys = top1s[ref_name]["query_key"].tolist()

    match_by_desc = (
        "identity (gallery_id)"
        if args.match_by == "id"
        else "gallery image (gallery_path)"
    )
    print(f"Models: {model_names}")
    print(f"Query images: {len(query_keys)}")
    print(f"Agreement basis: {match_by_desc}")
    print()

    n_agree = 0
    n_agree_correct = 0
    n_agree_wrong = 0
    n_disagree = 0
    n_disagree_any_correct = 0
    agreed_query_keys = []
    disagree_correct = {name: 0 for name in model_names}
    disagree_total = {name: 0 for name in model_names}
    disagree_candidates = {name: [] for name in model_names}
    disagree_min_first_hit_ranks = []
    disagree_combined_precision_sum = {k: 0.0 for k in args.topk}
    model_pairs = list(combinations(model_names, 2))
    pair_agree = {pair: 0 for pair in model_pairs}
    pair_agree_correct = {pair: 0 for pair in model_pairs}
    pair_total = {pair: 0 for pair in model_pairs}
    agreement_subset_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"n_total": 0, "n_agree": 0, "n_agree_correct": 0, "n_agree_wrong": 0}
    )

    print(f"{'Query ID':>10}  ", end="")
    for name in model_names:
        print(f"{name:>20}  ", end="")
    print(f"{'Agree?':>8}  {'Correct?':>10}")
    print("-" * (12 + 22 * len(model_names) + 22))

    for qkey in query_keys:
        predictions = {}
        match_keys = {}
        query_id = None
        for name in model_names:
            row = top1s[name][top1s[name]["query_key"] == qkey]
            if row.empty:
                predictions[name] = None
                match_keys[name] = None
                continue
            predictions[name] = int(row["gallery_id"].iloc[0])
            match_keys[name] = row["match_key"].iloc[0]
            if query_id is None:
                query_id = int(row["query_id"].iloc[0])

        valid_preds = [v for v in predictions.values() if v is not None]
        valid_match_keys = [v for v in match_keys.values() if v is not None]
        if not valid_preds:
            continue

        all_agree = len(set(valid_match_keys)) == 1
        is_correct = valid_preds[0] == query_id if all_agree else None
        any_correct = any(p == query_id for p in valid_preds)

        subset_stats = agreement_subset_stats[extract_subset_from_path(qkey)]
        subset_stats["n_total"] += 1

        if all_agree:
            n_agree += 1
            agreed_query_keys.append(qkey)
            subset_stats["n_agree"] += 1
            if is_correct:
                n_agree_correct += 1
                subset_stats["n_agree_correct"] += 1
            else:
                n_agree_wrong += 1
                subset_stats["n_agree_wrong"] += 1
        else:
            n_disagree += 1
            if any_correct:
                n_disagree_any_correct += 1
            this_query_candidates = {}
            for name in model_names:
                p = predictions[name]
                if p is None:
                    continue
                disagree_total[name] += 1
                if p == query_id:
                    disagree_correct[name] += 1
                q_candidates = candidate_lookup[name].get(qkey, [])
                this_query_candidates[name] = q_candidates
                disagree_candidates[name].append(q_candidates)

            min_first_hit = float("nan")
            for q_candidates in this_query_candidates.values():
                correct_ranks = [r for r, _, correct in q_candidates if correct]
                if correct_ranks:
                    fh = min(correct_ranks)
                    if pd.isna(min_first_hit) or fh < min_first_hit:
                        min_first_hit = fh
            disagree_min_first_hit_ranks.append(min_first_hit)

            for k in args.topk:
                # pooled window: union of each model's top-k candidates for this query
                pooled = {}
                for q_candidates in this_query_candidates.values():
                    for _, key, correct in q_candidates[:k]:
                        pooled.setdefault(key, correct)
                if pooled:
                    n_hit = sum(1 for correct in pooled.values() if correct)
                    precision = n_hit / len(pooled)
                else:
                    precision = 0.0
                disagree_combined_precision_sum[k] += precision
            for a, b in model_pairs:
                pa, pb = predictions[a], predictions[b]
                ka, kb = match_keys[a], match_keys[b]
                if pa is None or pb is None:
                    continue
                pair_total[(a, b)] += 1
                if ka == kb:
                    pair_agree[(a, b)] += 1
                    if pa == query_id:
                        pair_agree_correct[(a, b)] += 1

        print(f"{query_id:>10}  ", end="")
        for name in model_names:
            p = predictions[name]
            if p is None:
                print(f"{'N/A':>20}  ", end="")
            else:
                marker = " ✓" if p == query_id else ""
                print(f"{p:>18}{marker:>2}  ", end="")

        if all_agree:
            correct_str = "✓ YES" if is_correct else "✗ NO"
            print(f"{'YES':>8}  {correct_str:>10}")
        else:
            print(f"{'NO':>8}  {'---':>10}")

    total = n_agree + n_disagree
    print()
    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Total query images:             {total}")
    print(
        f"All models agree (top-1):       {n_agree}/{total} ({100*n_agree/total:.1f}%)"
    )
    if n_agree > 0:
        print(
            f"  → Correct when agreeing:      {n_agree_correct}/{n_agree} ({100*n_agree_correct/n_agree:.1f}%)"
        )
        print(
            f"  → Wrong when agreeing:        {n_agree_wrong}/{n_agree} ({100*n_agree_wrong/n_agree:.1f}%)"
        )
    if args.per_subset:
        print_per_subset_agreement(agreement_subset_stats)
    print(
        f"Models disagree (top-1):        {n_disagree}/{total} ({100*n_disagree/total:.1f}%)"
    )
    if n_disagree > 0:
        print(
            f"  → At least one correct:       {n_disagree_any_correct}/{n_disagree} ({100*n_disagree_any_correct/n_disagree:.1f}%)"
        )
        for k in args.topk:
            precision = disagree_combined_precision_sum[k] / n_disagree
            print(
                f"  → Combined precision@{k} (any model, pooled): {100*precision:.1f}% (n={n_disagree})"
            )
        print(f"  → Combined CMC (any model correct):")
        print_cmc_at_k_values(
            pd.Series(disagree_min_first_hit_ranks), args.topk, indent=28
        )

    print()
    print("Individual model top-1 accuracy:")
    for name in model_names:
        df = top1s[name]
        correct = (df["query_id"] == df["gallery_id"]).sum()
        print(f"  {name:>30}: {correct}/{len(df)} ({100*correct/len(df):.1f}%)")
        model_candidates = candidates[name]["candidates"]
        for k in args.topk:
            precision, n = precision_at_k(model_candidates, k)
            print(f"  {f'  → precision@{k}':>30}: {100*precision:.1f}% (n={n})")
        print_cmc_at_k_values(
            first_hit_rank_series(model_candidates), args.topk
        )
        if args.per_subset:
            rows = per_subset_top1_metrics(
                top1s[name], candidates[name], args.topk
            )
            print_per_subset_top1_metrics(rows, args.topk)

    print()
    print("Individual model top-1 accuracy on disagreement subset:")
    for name in model_names:
        c, t = disagree_correct[name], disagree_total[name]
        if t > 0:
            print(f"  {name:>30}: {c}/{t} ({100*c/t:.1f}%)")
        else:
            print(f"  {name:>30}: N/A (0 queries)")
        model_disagree_candidates = pd.Series(disagree_candidates[name])
        for k in args.topk:
            label = f"  → precision@{k}"
            if t > 0:
                precision, n = precision_at_k(model_disagree_candidates, k)
                print(f"  {label:>30}: {100*precision:.1f}% (n={n})")
            else:
                print(f"  {label:>30}: N/A (0 queries)")
        if t > 0:
            print_cmc_at_k_values(
                first_hit_rank_series(model_disagree_candidates), args.topk
            )

    print()
    print("Pairwise top-1 agreement on disagreement subset:")
    for a, b in model_pairs:
        c, t = pair_agree[(a, b)], pair_total[(a, b)]
        pair_label = f"{a} vs {b}"
        if t > 0:
            print(f"  {pair_label:>60}: {c}/{t} ({100*c/t:.1f}%)")
            if c > 0:
                cc = pair_agree_correct[(a, b)]
                print(
                    f"  {'':>60}    → correct when agreeing: {cc}/{c} ({100*cc/c:.1f}%)"
                )
        else:
            print(f"  {pair_label:>60}: N/A (0 queries)")

    if args.output and agreed_query_keys:
        source_path = args.source_csv if args.source_csv else args.results[0]
        source_df = pd.read_csv(source_path)
        source_df["query_key"] = source_df["query_path"].apply(normalize_path)
        agreed_set = set(agreed_query_keys)
        out_df = source_df[source_df["query_key"].isin(agreed_set)].drop(
            columns=["query_key"]
        )
        out_df.to_csv(args.output, index=False)
        print(
            f"\nSaved {len(out_df)} rows ({len(agreed_set)} agreed queries) → {args.output}"
        )
        print(f"  Source CSV: {source_path}")


if __name__ == "__main__":
    main()
