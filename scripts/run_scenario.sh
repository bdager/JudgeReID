#!/usr/bin/env bash
# Run the JudgeReID pipeline on one CHIRLA scenario (the steps in docs/reproduce.md)
#
# Usage
#   bash scripts/run_scenario.sh long_term
#   K=3 bash scripts/run_scenario.sh long_term              # P = 2K = 6
#   STEPS="gate judge final" bash scripts/run_scenario.sh long_term
#
# STEPS can list experts (embeddings and rankings), gate (agreement and candidate
# union), judge (VLM re-ranking, needs the vLLM server from run_vllm_server.sh) and final
# Finished outputs are skipped, so the script can be re-run after an interruption
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

SCENARIO="${1:?usage: run_scenario.sh <reappearance|long_term|multi_camera|multi_camera_long_term>}"
K="${K:-8}"
STEPS="${STEPS:-experts gate judge final}"
VLM="${VLM:-qwen3.6-27b}"
CONCURRENCY="${CONCURRENCY:-70}"
DIFFER_CKPT="${DIFFER_CKPT:-checkpoints/DIFFER/Models/Celeb_light}"
SAPIENSID_CKPT="${SAPIENSID_CKPT:-checkpoints/SapiensID/sapiensid_wb12m}"

if [[ -f .env ]]; then set -a; source .env; set +a; fi
: "${CHIRLA_ROOT:?set CHIRLA_ROOT in .env}"
: "${EXPERIMENTS_ROOT:?set EXPERIMENTS_ROOT in .env}"

EMB="$EXPERIMENTS_ROOT/embeddings"
GATE="$EXPERIMENTS_ROOT/gate/$SCENARIO"
RUN="$EXPERIMENTS_ROOT/rerank/$VLM/$SCENARIO/t$((2 * K))"
has_step() { [[ " $STEPS " == *" $1 "* ]]; }
log() { echo "[$SCENARIO] $*"; }

if has_step experts; then
    for SPLIT in gallery query; do
        CSV="$CHIRLA_ROOT/metadata/reid_${SCENARIO}_${SPLIT}.csv"
        # create_embeddings.py skips files that already exist
        python create_embeddings.py --csv "$CSV" --input "$CHIRLA_ROOT/benchmark" \
            --output "$EMB/differ" --model differ \
            --config "$DIFFER_CKPT/config.yml" --cktp "$DIFFER_CKPT/eva02_l_bio_best.pth"
        python create_embeddings.py --csv "$CSV" --input "$CHIRLA_ROOT/benchmark" \
            --output "$EMB/sapiensid" --model sapiensid --cktp "$SAPIENSID_CKPT"
    done
    for M in differ sapiensid; do
        if [[ -f "$EMB/$M/${SCENARIO}_results.csv" ]]; then
            log "skip $M rankings (exist)"
            continue
        fi
        python metrics/evaluate_reid.py \
            --gallery "$EMB/$M/${SCENARIO}_gallery_embeddings.h5" \
            --query "$EMB/$M/${SCENARIO}_query_embeddings.h5" \
            --topk 1 3 5 10 20 --per-subset \
            --out-csv "$EMB/$M/${SCENARIO}_results.csv"
    done
fi

if has_step gate; then
    mkdir -p "$GATE"
    if [[ ! -f "$GATE/agreed.csv" ]]; then
        python ensemble/top1_agreement.py \
            --results "$EMB/differ/${SCENARIO}_results.csv" "$EMB/sapiensid/${SCENARIO}_results.csv" \
            --names differ sapiensid \
            --source-csv "$EMB/sapiensid/${SCENARIO}_results.csv" \
            --per-subset --output "$GATE/agreed.csv"
    fi
    for M in differ sapiensid; do
        [[ -f "$GATE/${M}_disagreed.csv" ]] || python ensemble/remove_matching_queries.py \
            --ref "$GATE/agreed.csv" --base "$EMB/$M/${SCENARIO}_results.csv" \
            --output "$GATE/${M}_disagreed.csv"
    done
    [[ -f "$GATE/union_k$K.csv" ]] || python ensemble/merge_topk.py --k "$K" \
        --results "$GATE/differ_disagreed.csv" "$GATE/sapiensid_disagreed.csv" \
        --names differ sapiensid --output "$GATE/union_k$K.csv"
fi

if has_step judge; then
    # Queries already in $RUN/results.csv are skipped, 
    # the judge resumes on its own
    python eval_vlm_rerank.py \
        --results_csv "$GATE/union_k$K.csv" \
        --topk $((2 * K)) --vlm "$VLM" \
        --sys_prompt reasoning/prompts/sys_rerank.txt \
        --data_root "$CHIRLA_ROOT/benchmark/" \
        --label_images --process_all --retry_fallbacks \
        --concurrency "$CONCURRENCY" \
        --output_dir "$RUN"
fi

if has_step final; then
    python ensemble/build_final_results.py \
        --results "$EMB/differ/${SCENARIO}_results.csv" "$EMB/sapiensid/${SCENARIO}_results.csv" \
        --names differ sapiensid \
        --agree-csv "$GATE/agreed.csv" \
        --candidates-csv "$GATE/union_k$K.csv" \
        --rerank-csv "$RUN/results.csv" \
        --merge-k "$K" --head-size "$K" \
        --shared-credit all --shared-credit-tiebreak sapiensid --fallback-model sapiensid \
        --topk 1 3 5 10 20 --per-subset \
        --output-dir "$RUN/final"
    log "done: $RUN/final/final_report.json"
fi
