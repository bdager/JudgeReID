#!/usr/bin/env python3
"""
Ensemble re-ID methods from multiple H5 embedding files.

Supports three combination strategies:
  voting       – Borda count: each model contributes (N - rank) points per gallery item
  score_fusion – min-max normalize similarity scores per model, then average
  rrf          – Reciprocal Rank Fusion: sum of 1 / (k + rank) across models

Works on raw embeddings to avoid the top-K truncation of pre-saved CSVs.
Gallery and query items are aligned across models by normalized path suffix.

Usage:
    python ensemble/ensemble.py \
        --query  model_a/long_term_query_embeddings.h5   model_b/long_term_query_embeddings.h5 \
        --gallery model_a/long_term_gallery_embeddings.h5 model_b/long_term_gallery_embeddings.h5 \
        --method voting score_fusion rrf \
        --output-dir /path/to/output/ \
        --save-topk 20
"""

import argparse
import csv
import json
import os
import re
from pathlib import Path

import h5py
import numpy as np


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def normalize_path(p: str) -> str:
    """Strip absolute prefix, keep reid/long_term/... suffix for cross-model alignment."""
    m = re.search(r"(reid/[^/]+/.+)$", p)
    return m.group(1) if m else p


def load_h5(path: str):
    with h5py.File(path, "r") as f:
        embeddings = f["embeddings"][:]
        ids = f["ids"][:]
        paths = np.array([normalize_path(p) for p in f["paths"][:].astype(str)])
    return embeddings, ids, paths


def cosine_sim(query_emb: np.ndarray, gallery_emb: np.ndarray) -> np.ndarray:
    q = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)
    g = gallery_emb / np.linalg.norm(gallery_emb, axis=1, keepdims=True)
    return np.dot(q, g.T)


def write_results_csv(
    output_path: str,
    query_paths, query_ids,
    gallery_paths, gallery_ids,
    scores: np.ndarray,
    save_topk: int,
):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_path", "query_id", "gallery_path", "gallery_id", "rank", "score"])
        for i in range(len(query_paths)):
            top_idx = np.argsort(-scores[i])[:save_topk]
            for rank, j in enumerate(top_idx, start=1):
                writer.writerow([
                    query_paths[i], int(query_ids[i]),
                    gallery_paths[j], int(gallery_ids[j]),
                    rank, float(scores[i, j]),
                ])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_cmc_map(scores: np.ndarray, query_ids, gallery_ids, topk_values, trunc_topk=None):
    """Compute CMC@K (always over the full ranking) and mAP.

    trunc_topk: if set, mAP is computed only over each query's top-`trunc_topk`
    ranked gallery items (matching relevant items beyond that rank are treated
    as not retrieved, same bias a truncated top-K results CSV would have). If
    None (default), mAP uses the full ranking.
    """
    n_queries = scores.shape[0]
    cmc = np.zeros(len(topk_values))
    aps = []

    for i in range(n_queries):
        sort_idx = np.argsort(-scores[i])
        sorted_gids = gallery_ids[sort_idx]
        matches = (sorted_gids == query_ids[i]).astype(int)

        for j, k in enumerate(topk_values):
            if matches[:k].sum() > 0:
                cmc[j] += 1

        ap_matches = matches[:trunc_topk] if trunc_topk is not None else matches
        n_gt = ap_matches.sum()
        if n_gt > 0:
            hits, sum_p = 0, 0.0
            for pos, m in enumerate(ap_matches):
                if m:
                    hits += 1
                    sum_p += hits / (pos + 1)
            aps.append(sum_p / n_gt)
        else:
            aps.append(0.0)

    return cmc / n_queries, float(np.mean(aps))


def format_metrics_line(label: str, cmc, mAP: float, topk_values) -> str:
    parts = "  ".join(f"CMC@{k}: {v*100:.2f}%" for k, v in zip(topk_values, cmc))
    return f"[{label}]  {parts}  mAP: {mAP*100:.2f}%"


def print_metrics(label: str, scores, query_ids, gallery_ids, topk_values, trunc_topk=None) -> str:
    cmc, mAP = compute_cmc_map(scores, query_ids, gallery_ids, topk_values, trunc_topk=trunc_topk)
    line = format_metrics_line(label, cmc, mAP, topk_values)
    print(line)
    return line


# ---------------------------------------------------------------------------
# Per-subset metrics (test_N queries vs. corresponding train_N gallery)
# ---------------------------------------------------------------------------

def extract_subset_from_path(path: str) -> str:
    """Extract 'test_N' / 'train_N' from a CHIRLA image path, e.g.
    '.../reid/long_term/test/test_2/seq_002/.../frame_2657.png' -> 'test_2'."""
    parts = path.split("/")
    for i, part in enumerate(parts):
        if part in ("test", "train") and i + 1 < len(parts):
            next_part = parts[i + 1]
            if next_part.startswith(("test_", "train_")):
                return next_part
    return "unknown"


def group_indices_by_subset(paths) -> dict:
    """Return {subset_name: np.ndarray of indices into `paths`}."""
    groups = {}
    for i, p in enumerate(paths):
        groups.setdefault(extract_subset_from_path(p), []).append(i)
    return {name: np.array(idx) for name, idx in groups.items()}


def compute_cmc_map_per_subset(
    scores: np.ndarray, query_ids, query_paths, gallery_ids, gallery_paths,
    topk_values, trunc_topk=None,
):
    """Split by query subset (test_N), match against the corresponding gallery
    subset (train_N) when present (else fall back to the full gallery), and
    average CMC/mAP with equal weight per subset.

    Returns (avg_cmc, avg_mAP, subset_results) where subset_results maps
    subset_name -> {"cmc", "mAP", "num_queries"}.
    """
    query_subsets = group_indices_by_subset(query_paths)
    gallery_subsets = group_indices_by_subset(gallery_paths)

    subset_results = {}
    for qsubset, q_idx in query_subsets.items():
        g_idx = gallery_subsets.get(qsubset.replace("test_", "train_"))
        if g_idx is None:
            g_idx = np.arange(len(gallery_ids))
        sub_scores = scores[np.ix_(q_idx, g_idx)]
        cmc, mAP = compute_cmc_map(
            sub_scores, query_ids[q_idx], gallery_ids[g_idx], topk_values, trunc_topk=trunc_topk
        )
        subset_results[qsubset] = {"cmc": cmc, "mAP": mAP, "num_queries": len(q_idx)}

    if subset_results:
        avg_cmc = np.mean([r["cmc"] for r in subset_results.values()], axis=0)
        avg_mAP = float(np.mean([r["mAP"] for r in subset_results.values()]))
    else:
        avg_cmc, avg_mAP = np.zeros(len(topk_values)), 0.0
    return avg_cmc, avg_mAP, subset_results


def cmc_map_from_ranked_lists(query_ids, ranked_lists, topk_values):
    """CMC/mAP from per-query ranked gallery-ID lists (CSV/RRF mode, no score matrix)."""
    cmc = np.zeros(len(topk_values))
    aps = []
    for qid, ranked in zip(query_ids, ranked_lists):
        matches = np.array([1 if gid == qid else 0 for gid in ranked])
        for j, k in enumerate(topk_values):
            if matches[:k].sum() > 0:
                cmc[j] += 1
        n_gt = matches.sum()
        if n_gt > 0:
            hits, sum_p = 0, 0.0
            for pos, m in enumerate(matches):
                if m:
                    hits += 1
                    sum_p += hits / (pos + 1)
            aps.append(sum_p / n_gt)
        else:
            aps.append(0.0)
    cmc /= len(query_ids)
    return cmc, float(np.mean(aps))


def cmc_map_from_ranked_lists_per_subset(query_ids, query_paths, ranked_lists, topk_values):
    """Per-subset (test_N) version of cmc_map_from_ranked_lists, equal-weight averaged."""
    query_ids = np.asarray(query_ids)
    subsets = group_indices_by_subset(query_paths)

    subset_results = {}
    for name, idx in subsets.items():
        cmc, mAP = cmc_map_from_ranked_lists(query_ids[idx], [ranked_lists[i] for i in idx], topk_values)
        subset_results[name] = {"cmc": cmc, "mAP": mAP, "num_queries": len(idx)}

    if subset_results:
        avg_cmc = np.mean([r["cmc"] for r in subset_results.values()], axis=0)
        avg_mAP = float(np.mean([r["mAP"] for r in subset_results.values()]))
    else:
        avg_cmc, avg_mAP = np.zeros(len(topk_values)), 0.0
    return avg_cmc, avg_mAP, subset_results


def report_per_subset(label: str, subset_results: dict, topk_values) -> list:
    """Print and return formatted lines for a subset-average plus per-subset breakdown."""
    lines = []
    for name in sorted(subset_results):
        r = subset_results[name]
        line = "    " + format_metrics_line(f"{label} / {name} ({r['num_queries']} q)", r["cmc"], r["mAP"], topk_values)
        print(line)
        lines.append(line)
    return lines


# ---------------------------------------------------------------------------
# Ensemble methods
# ---------------------------------------------------------------------------

def minmax_normalize(sim: np.ndarray) -> np.ndarray:
    """Normalize each query's score vector to [0, 1]."""
    mins = sim.min(axis=1, keepdims=True)
    maxs = sim.max(axis=1, keepdims=True)
    denom = np.where(maxs - mins == 0, 1.0, maxs - mins)
    return (sim - mins) / denom


def score_fusion(sim_matrices: list) -> np.ndarray:
    """Average of per-model min-max normalized similarity scores."""
    return np.mean([minmax_normalize(sim) for sim in sim_matrices], axis=0)


def rrf(sim_matrices: list, k: int = 60) -> np.ndarray:
    """Reciprocal Rank Fusion: score(item) = sum_models 1 / (k + rank)."""
    rrf_scores = np.zeros_like(sim_matrices[0])
    for sim in sim_matrices:
        # argsort of argsort gives rank of each item (0 = best)
        ranks = np.argsort(np.argsort(-sim, axis=1), axis=1)
        rrf_scores += 1.0 / (k + ranks + 1)
    return rrf_scores


def borda(sim_matrices: list) -> np.ndarray:
    """Borda count: each model gives (N - 1 - rank) points to each gallery item."""
    n_gallery = sim_matrices[0].shape[1]
    borda_scores = np.zeros_like(sim_matrices[0])
    for sim in sim_matrices:
        ranks = np.argsort(np.argsort(-sim, axis=1), axis=1)
        borda_scores += (n_gallery - 1 - ranks)
    return borda_scores


def rrf_from_csv(
    csv_paths: list,
    model_names: list,
    k: int = 60,
    penalty_rank: int = None,
    save_topk: int = 20,
    open_map_threshold: float = None,
    original_results: str = None,
):
    """
    Reciprocal Rank Fusion directly from results CSVs (no embeddings needed).

    Gallery items present in some models but absent in others receive a penalty
    rank (default: max rank saved in the CSV + 1, typically 21 for top-20 CSVs).

    original_results: path to the base embeddings CSV (gallery_path, gallery_id, rank).
      When provided, each rerank-format model's ranked list is extended with the
      original items not already in the VLM window, giving a fuller list for RRF
      and more accurate metric computation.
    """
    import pandas as pd

    # Load original results once if provided
    original_rankings  = {}  # query_key -> {gallery_key: rank}  (0-indexed after VLM window)
    original_gallery_meta = {}  # gallery_key -> (gallery_id, gallery_path)

    if original_results:
        orig_df = pd.read_csv(original_results)
        if open_map_threshold is None:
            orig_df = orig_df[orig_df["query_id"] >= 0].reset_index(drop=True)
        orig_df["query_key"]   = orig_df["query_path"].apply(normalize_path)
        orig_df["gallery_key"] = orig_df["gallery_path"].apply(normalize_path)
        for qkey, group in orig_df.groupby("query_key"):
            original_rankings[qkey] = {
                row["gallery_key"]: int(row["rank"])
                for _, row in group.iterrows()
            }
            for _, row in group.iterrows():
                gkey = row["gallery_key"]
                if gkey not in original_gallery_meta:
                    original_gallery_meta[gkey] = (int(row["gallery_id"]), row["gallery_path"])
        print(f"  Original results loaded: {len(original_rankings)} queries, up to rank {orig_df['rank'].max()}")

    def _load_rankings(path):
        """
        Return a dict: query_key -> {gallery_key: rank}.

        Supports two CSV formats:
          - Embeddings format: query_path, query_id, gallery_path, gallery_id, rank, score
          - Rerank format:     query_path, query_id, reranked_gallery_ids (JSON list of IDs)
        """
        df = pd.read_csv(path)
        if open_map_threshold is None:
            df = df[df["query_id"] >= 0].reset_index(drop=True)
        df["query_key"] = df["query_path"].apply(normalize_path)

        rankings   = {}  # query_key -> {gallery_key: rank}
        query_info = {}  # query_key -> (query_id, query_path)
        gallery_meta_out = {}  # gallery_key -> (gallery_id, gallery_path or None)

        if "reranked_gallery_ids" in df.columns:
            # Rerank format: one row per query, gallery IDs as a JSON list
            for _, row in df.iterrows():
                qkey   = row["query_key"]
                qid    = int(row["query_id"])
                ids    = json.loads(row["reranked_gallery_ids"])
                ranking = {str(gid): r for r, gid in enumerate(ids, start=1)}
                rankings[qkey]   = ranking
                query_info[qkey] = (qid, row["query_path"])
                for gid in ids:
                    gkey = str(gid)
                    if gkey not in gallery_meta_out:
                        gallery_meta_out[gkey] = (int(gid), None)
        else:
            # Embeddings format: one row per (query, gallery item)
            df["gallery_key"] = df["gallery_path"].apply(normalize_path)
            for qkey, group in df.groupby("query_key"):
                qid     = int(group["query_id"].iloc[0])
                ranking = {row["gallery_key"]: int(row["rank"]) for _, row in group.iterrows()}
                rankings[qkey]   = ranking
                query_info[qkey] = (qid, group["query_path"].iloc[0])
                for _, row in group.iterrows():
                    gkey = row["gallery_key"]
                    if gkey not in gallery_meta_out:
                        gallery_meta_out[gkey] = (int(row["gallery_id"]), row["gallery_path"])

        return rankings, query_info, gallery_meta_out

    all_rankings   = []
    all_query_info = {}
    all_gallery_meta = {}

    for name, path in zip(model_names, csv_paths):
        rankings, query_info, gallery_meta = _load_rankings(path)

        # Extend rerank-format rankings with original results
        if original_results:
            for qkey, ranking in rankings.items():
                orig = original_rankings.get(qkey, {})
                # append original items not already ranked, preserving original order
                next_rank = max(ranking.values(), default=0) + 1
                for gkey, _ in sorted(orig.items(), key=lambda x: x[1]):
                    if gkey not in ranking:
                        ranking[gkey] = next_rank
                        next_rank += 1
            # pull in gallery_meta from original for any new keys
            gallery_meta.update({k: v for k, v in original_gallery_meta.items() if k not in gallery_meta})

        all_rankings.append(rankings)
        all_query_info.update(query_info)
        all_gallery_meta.update(gallery_meta)

    if penalty_rank is None:
        max_rank = max(max(r.values()) for rankings in all_rankings for r in rankings.values() if r)
        penalty_rank = int(max_rank) + 1
    print(f"  Penalty rank for missing items: {penalty_rank}")

    ref_query_keys = list(all_rankings[0].keys())

    rows = []
    query_ranked_ids = {}

    for qkey in ref_query_keys:
        query_id, query_path = all_query_info[qkey]

        all_gallery_keys = set().union(*(r.get(qkey, {}).keys() for r in all_rankings))

        item_scores = {
            gkey: sum(
                1.0 / (k + rankings.get(qkey, {}).get(gkey, penalty_rank))
                for rankings in all_rankings
            )
            for gkey in all_gallery_keys
        }

        sorted_items = sorted(item_scores.items(), key=lambda x: -x[1])
        query_ranked_ids[qkey] = [all_gallery_meta[gkey][0] for gkey, _ in sorted_items]

        for rank, (gkey, score) in enumerate(sorted_items[:save_topk], start=1):
            gallery_id, gpath = all_gallery_meta[gkey]
            row = {
                "query_path": query_path,
                "query_id":   query_id,
                "gallery_id": gallery_id,
                "rank":       rank,
                "score":      score,
            }
            if gpath is not None:
                row["gallery_path"] = gpath
            rows.append(row)

    return pd.DataFrame(rows), query_ranked_ids


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

def align_embeddings(ref_paths, paths, embeddings):
    """Reorder embeddings to match ref_paths ordering."""
    path_to_idx = {p: i for i, p in enumerate(paths)}
    missing = [p for p in ref_paths if p not in path_to_idx]
    if missing:
        raise ValueError(f"{len(missing)} paths from reference not found in this model. First missing: {missing[0]}")
    order = [path_to_idx[p] for p in ref_paths]
    return embeddings[order]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Ensemble re-ID from H5 embedding files or pre-saved results CSVs.")
    parser.add_argument("--query",       nargs="+", default=None, help="Query H5 files, one per model (H5 mode).")
    parser.add_argument("--gallery",     nargs="+", default=None, help="Gallery H5 files, one per model (H5 mode).")
    parser.add_argument("--results-csv", nargs="+", default=None, dest="results_csv",
                        help="Pre-saved results CSVs, one per model. Enables CSV-based RRF (no embeddings needed).")
    parser.add_argument(
        "--method", nargs="+", default=["rrf"],
        choices=["voting", "score_fusion", "rrf"],
        help="Ensemble method(s) to apply (default: rrf).",
    )
    parser.add_argument("--names", nargs="+", default=None, help="Model names (auto-derived if omitted).")
    parser.add_argument("--output-dir", dest="output_dir", required=True, help="Directory to save result CSVs.")
    parser.add_argument("--save-topk", dest="save_topk", type=int, default=20, help="Ranks to save per query (default: 20).")
    parser.add_argument("--topk", nargs="+", type=int, default=[1, 5, 10], help="K values for CMC metrics.")
    parser.add_argument(
        "--trunc-topk", dest="trunc_topk", type=int, default=None,
        help="If set, compute mAP over only each query's top-K ranked gallery items "
             "(matching relevant items beyond rank K are ignored). Default: use the "
             "full ranking, giving an exact mAP. CMC@k is unaffected by this flag.",
    )
    parser.add_argument("--rrf-k", dest="rrf_k", type=int, default=60, help="RRF constant k (default: 60).")
    parser.add_argument(
        "--penalty-rank", dest="penalty_rank", type=int, default=None,
        help="CSV mode only. Rank assigned to gallery items absent from a model's list "
             "(default: max rank in CSV + 1, typically 21 for top-20 CSVs).",
    )
    parser.add_argument(
        "--original-results", dest="original_results", default=None,
        help="CSV mode only. Path to the base embeddings results CSV (gallery_path, gallery_id, rank). "
             "Extends each rerank-format model's ranked list with original items beyond the VLM window, "
             "giving a fuller ranking and more accurate metrics.",
    )
    parser.add_argument(
        "--open-map-threshold", dest="open_map_threshold", type=float, default=None,
        help="When set, include queries with negative IDs. By default they are excluded.",
    )
    parser.add_argument(
        "--per-subset", dest="per_subset", action="store_true",
        help="Also compute CMC/mAP per test_N subset (query test_N vs. gallery train_N, "
             "equal-weight averaged), in addition to the pooled metrics. Written to "
             "<output-dir>/metrics.txt alongside the console output.",
    )
    args = parser.parse_args()

    # Validate: exactly one of H5 mode or CSV mode must be used
    if args.results_csv and (args.query or args.gallery):
        parser.error("Use either --results-csv OR --query/--gallery, not both.")
    if not args.results_csv and not (args.query and args.gallery):
        parser.error("Provide either --results-csv (CSV mode) or both --query and --gallery (H5 mode).")
    if args.query and args.gallery:
        assert len(args.query) == len(args.gallery), "--query and --gallery must have the same number of files."

    source_paths = args.results_csv if args.results_csv else args.query
    if args.names:
        assert len(args.names) == len(source_paths), "--names must match number of models."
        model_names = args.names
    else:
        model_names = ["/".join(Path(p).parts[-3:-1]) for p in source_paths]

    # -----------------------------------------------------------------------
    # CSV mode: RRF from pre-saved results CSVs
    # -----------------------------------------------------------------------
    if args.results_csv:
        if set(args.method) - {"rrf"}:
            print("Note: CSV mode only supports rrf. Ignoring other methods.")

        os.makedirs(args.output_dir, exist_ok=True)
        print(f"CSV-based RRF over {len(args.results_csv)} models: {model_names}")

        out_df, query_ranked_ids = rrf_from_csv(
            args.results_csv,
            model_names,
            k=args.rrf_k,
            penalty_rank=args.penalty_rank,
            save_topk=args.save_topk,
            open_map_threshold=args.open_map_threshold,
            original_results=args.original_results,
        )

        out_path = os.path.join(args.output_dir, "results_rrf_csv.csv")
        out_df.to_csv(out_path, index=False)

        # Compute metrics from the ranked id lists
        query_paths_arr = np.array(list(query_ranked_ids.keys()))
        query_ids    = np.array([out_df[out_df["query_path"].apply(normalize_path) == qk]["query_id"].iloc[0]
                                 for qk in query_ranked_ids])
        ranked_lists = [query_ranked_ids[qk] for qk in query_ranked_ids]

        cmc, mAP = cmc_map_from_ranked_lists(query_ids, ranked_lists, args.topk)
        line = format_metrics_line("rrf_csv", cmc, mAP, args.topk)
        line += f"  (mAP is approximate — truncated to union of top-{args.save_topk})"
        print(line)
        metrics_lines = [line]

        if args.per_subset:
            avg_cmc, avg_mAP, subset_results = cmc_map_from_ranked_lists_per_subset(
                query_ids, query_paths_arr, ranked_lists, args.topk
            )
            avg_line = format_metrics_line("rrf_csv (subset avg)", avg_cmc, avg_mAP, args.topk)
            print(avg_line)
            metrics_lines.append(avg_line)
            metrics_lines.extend(report_per_subset("rrf_csv", subset_results, args.topk))

        print(f"  Saved → {out_path}")

        metrics_path = os.path.join(args.output_dir, "metrics.txt")
        with open(metrics_path, "w") as f:
            f.write("\n".join(metrics_lines) + "\n")
        print(f"  Metrics saved → {metrics_path}")
        return

    # Load all models
    print("Loading embeddings...")
    query_data, gallery_data = [], []
    for name, qpath, gpath in zip(model_names, args.query, args.gallery):
        q_emb, q_ids, q_paths = load_h5(qpath)
        g_emb, g_ids, g_paths = load_h5(gpath)
        query_data.append((name, q_emb, q_ids, q_paths))
        gallery_data.append((name, g_emb, g_ids, g_paths))
        print(f"  {name}: query={q_emb.shape}, gallery={g_emb.shape}")

    # Use first model as reference for path ordering
    ref_q_paths = query_data[0][3]
    ref_g_paths = gallery_data[0][3]
    ref_q_ids   = query_data[0][2]
    ref_g_ids   = gallery_data[0][2]

    # Align all models to reference ordering
    print("Aligning paths across models...")
    aligned_q_embs, aligned_g_embs = [], []
    for (name, q_emb, q_ids, q_paths), (_, g_emb, g_ids, g_paths) in zip(query_data, gallery_data):
        aligned_q_embs.append(align_embeddings(ref_q_paths, q_paths, q_emb))
        aligned_g_embs.append(align_embeddings(ref_g_paths, g_paths, g_emb))

    # Filter out unknown (negative ID) queries
    if args.open_map_threshold is None:
        known_mask = ref_q_ids >= 0
        n_filtered = (~known_mask).sum()
        if n_filtered > 0:
            print(f"Filtered {n_filtered} unknown (negative ID) queries.")
        query_paths = ref_q_paths[known_mask]
        query_ids   = ref_q_ids[known_mask]
        aligned_q_embs = [e[known_mask] for e in aligned_q_embs]
    else:
        query_paths = ref_q_paths
        query_ids   = ref_q_ids

    gallery_paths = ref_g_paths
    gallery_ids   = ref_g_ids

    # Compute per-model similarity matrices
    print(f"\nComputing similarities ({len(query_paths)} queries × {len(gallery_paths)} gallery)...")
    metrics_lines = []
    sim_matrices = []
    for name, q_emb, g_emb in zip(model_names, aligned_q_embs, aligned_g_embs):
        sim = cosine_sim(q_emb, g_emb)
        sim_matrices.append(sim)
        metrics_lines.append(print_metrics(f"  {name}", sim, query_ids, gallery_ids, args.topk, trunc_topk=args.trunc_topk))
        if args.per_subset:
            avg_cmc, avg_mAP, subset_results = compute_cmc_map_per_subset(
                sim, query_ids, query_paths, gallery_ids, gallery_paths, args.topk, trunc_topk=args.trunc_topk
            )
            avg_line = format_metrics_line(f"{name} (subset avg)", avg_cmc, avg_mAP, args.topk)
            print(f"  {avg_line}")
            metrics_lines.append(avg_line)
            metrics_lines.extend(report_per_subset(name, subset_results, args.topk))

    # Apply ensemble methods
    print()
    method_map = {
        "score_fusion": lambda: score_fusion(sim_matrices),
        "rrf":          lambda: rrf(sim_matrices, k=args.rrf_k),
        "voting":       lambda: borda(sim_matrices),
    }

    os.makedirs(args.output_dir, exist_ok=True)

    for method in args.method:
        scores = method_map[method]()
        out_path = os.path.join(args.output_dir, f"results_{method}.csv")
        write_results_csv(out_path, query_paths, query_ids, gallery_paths, gallery_ids, scores, args.save_topk)
        metrics_lines.append(print_metrics(method, scores, query_ids, gallery_ids, args.topk, trunc_topk=args.trunc_topk))
        if args.per_subset:
            avg_cmc, avg_mAP, subset_results = compute_cmc_map_per_subset(
                scores, query_ids, query_paths, gallery_ids, gallery_paths, args.topk, trunc_topk=args.trunc_topk
            )
            avg_line = format_metrics_line(f"{method} (subset avg)", avg_cmc, avg_mAP, args.topk)
            print(avg_line)
            metrics_lines.append(avg_line)
            metrics_lines.extend(report_per_subset(method, subset_results, args.topk))
        print(f"  Saved → {out_path}")

    metrics_path = os.path.join(args.output_dir, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("\n".join(metrics_lines) + "\n")
    print(f"\nMetrics saved → {metrics_path}")


if __name__ == "__main__":
    main()
