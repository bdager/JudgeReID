#!/usr/bin/env bash
# ── vLLM model server launcher ──────────────────────────────────────────
#
# Reads vllm_server settings from  configs/vlm/<model>.yaml  so every
# model's launch configuration lives next to its generation defaults.
#
# Usage:
#   bash run_vllm_server.sh <model_config_name> [--gpus 0,1,2,3] [--port PORT] [--served-model-name NAME]
#
# Examples:
#   bash run_vllm_server.sh qwen3.6-27b --gpus 0,1                  # configs/vlm/qwen3.6-27b.yaml
#   bash run_vllm_server.sh qwen3.6-35b                             # configs/vlm/qwen3.6-35b.yaml
#   bash run_vllm_server.sh qwen3.6-27b --port 9090                 # override port
#   bash run_vllm_server.sh qwen3.6-27b --served-model-name my-qwen # override served name
#
# Notes:
# - Lowering max-model-len cuts KV cache size.
# - expandable_segments can reduce CUDA fragmentation during load.
# - disable-custom-all-reduce fixes "custom_all_reduce.cuh:455 invalid
#   argument" error by using NCCL instead of vLLM's custom kernel.
# ─────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Same .env the Python scripts read (MODELS_DIR below).
if [[ -f "$SCRIPT_DIR/.env" ]]; then
    set -a; source "$SCRIPT_DIR/.env"; set +a
fi

# ── Parse arguments ─────────────────────────────────────────────────────
MODEL_NAME=""
CUDA_DEVICES=""
CLI_PORT=""
CLI_SERVED_NAME=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)              CUDA_DEVICES="$2"; shift 2 ;;
        --port)              CLI_PORT="$2"; shift 2 ;;
        --served-model-name) CLI_SERVED_NAME="$2"; shift 2 ;;
        --help|-h)
            sed -n '2,/^# ──────/p' "$0" | head -n -1
            exit 0
            ;;
        *)
            if [[ -z "$MODEL_NAME" ]]; then
                MODEL_NAME="$1"; shift
            else
                echo "Unknown argument: $1" >&2; exit 1
            fi
            ;;
    esac
done

if [[ -z "$MODEL_NAME" ]]; then
    echo "Usage: bash run_vllm_server.sh <model_config_name> [--gpus 0,1,2,3]" >&2
    echo "Available configs:" >&2
    ls "$SCRIPT_DIR/configs/vlm/"*.yaml 2>/dev/null | sed 's|.*/||;s|\.yaml$||' | sed 's/^/  /' >&2
    exit 1
fi

CONFIG_FILE="$SCRIPT_DIR/configs/vlm/${MODEL_NAME}.yaml"
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: config file not found: $CONFIG_FILE" >&2
    exit 1
fi

# ── Read YAML values (lightweight: grep + sed, no external deps) ────────
yaml_get() {
    # Usage: yaml_get <key> <file> [default]
    # Reads a top-level or vllm_server.* key from a simple YAML file.
    local key="$1" file="$2" default="${3:-}"
    local value
    value=$(grep -E "^\s+${key}:" "$file" 2>/dev/null | head -1 \
            | sed 's/^[^:]*:\s*//' | sed 's/\s*#.*//' | sed 's/^"\(.*\)"$/\1/' | sed "s/^'\(.*\)'$/\1/")
    if [[ -z "$value" ]]; then
        value=$(grep -E "^${key}:" "$file" 2>/dev/null | head -1 \
                | sed 's/^[^:]*:\s*//' | sed 's/\s*#.*//' | sed 's/^"\(.*\)"$/\1/' | sed "s/^'\(.*\)'$/\1/")
    fi
    echo "${value:-$default}"
}

# Read vllm_server section values
MODEL_PATH=$(yaml_get "model_path" "$CONFIG_FILE")
SERVED_NAME=$(yaml_get "served_model_name" "$CONFIG_FILE")
PORT=$(yaml_get "port" "$CONFIG_FILE" "8787")
TP_SIZE=$(yaml_get "tensor_parallel_size" "$CONFIG_FILE" "1")
GPU_MEM=$(yaml_get "gpu_memory_utilization" "$CONFIG_FILE" "0.90")
MAX_MODEL_LEN=$(yaml_get "max_model_len" "$CONFIG_FILE")
DISABLE_ALLREDUCE=$(yaml_get "disable_custom_all_reduce" "$CONFIG_FILE" "false")
SWAP_SPACE=$(yaml_get "swap_space" "$CONFIG_FILE" "")
EXTRA_ARGS=$(yaml_get "extra_args" "$CONFIG_FILE" "")
LIMIT_MM=$(yaml_get "extra_limit_mm" "$CONFIG_FILE" "")
EXTRA_ENV=$(yaml_get "extra_env" "$CONFIG_FILE" "")

if [[ -z "$MODEL_PATH" ]]; then
    echo "Error: 'vllm_server.model_path' not set in $CONFIG_FILE" >&2
    exit 1
fi
if [[ -z "$SERVED_NAME" ]]; then
    SERVED_NAME="$MODEL_NAME"
fi

# CLI overrides
[[ -n "$CLI_PORT" ]]        && PORT="$CLI_PORT"
[[ -n "$CLI_SERVED_NAME" ]] && SERVED_NAME="$CLI_SERVED_NAME"

# Configs name Hub ids (Qwen/Qwen3.6-27B) so a fresh clone works anywhere;
# a local copy under $MODELS_DIR avoids re-downloading the weights
if [[ -n "${MODELS_DIR:-}" && -d "$MODELS_DIR/$(basename "$MODEL_PATH")" ]]; then
    MODEL_PATH="$MODELS_DIR/$(basename "$MODEL_PATH")"
fi

# ── Build vllm serve command ────────────────────────────────────────────
VLLM_BIN=$(command -v vllm 2>/dev/null) || true
if [[ -z "$VLLM_BIN" ]]; then
    echo "Error: 'vllm' not found on PATH. Activate the correct conda env first." >&2
    exit 1
fi
CMD=("$VLLM_BIN" serve "$MODEL_PATH")
CMD+=(--port "$PORT")
CMD+=(--served-model-name "$SERVED_NAME")
CMD+=(--tensor-parallel-size "$TP_SIZE")
CMD+=(--gpu-memory-utilization "$GPU_MEM")

if [[ -n "$MAX_MODEL_LEN" ]]; then
    CMD+=(--max-model-len "$MAX_MODEL_LEN")
fi
if [[ "$DISABLE_ALLREDUCE" == "true" ]]; then
    CMD+=(--disable-custom-all-reduce)
fi
if [[ -n "$SWAP_SPACE" ]]; then
    CMD+=(--swap-space "$SWAP_SPACE")
fi

# Append any model-specific extra arguments
if [[ -n "$EXTRA_ARGS" ]]; then
    # shellcheck disable=SC2206
    CMD+=($EXTRA_ARGS)
fi
if [[ -n "$LIMIT_MM" ]]; then
    CMD+=(--limit-mm-per-prompt "$LIMIT_MM")
fi

# ── Environment variables ───────────────────────────────────────────────
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1

# flashinfer JIT needs nvcc; point CUDA_HOME at the conda env if not already set
if [[ -z "${CUDA_HOME:-}" && -z "${CUDA_PATH:-}" ]]; then
    export CUDA_HOME="${CONDA_PREFIX:-$(dirname "$(dirname "$(command -v nvcc)")")}"
fi
# flashinfer JIT linker needs to find -lcudart
export LIBRARY_PATH="${CUDA_HOME}/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"

if [[ -n "$CUDA_DEVICES" ]]; then
    export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
fi

# Model-specific environment variables (e.g. NCCL tuning)
if [[ -n "$EXTRA_ENV" ]]; then
    for kv in $EXTRA_ENV; do
        export "$kv"
    done
fi

# ── Logging ─────────────────────────────────────────────────────────────
# Screen scrollback is unreliable (gets overwritten by nvtop/other panes),
# so persist all server output to a file as well as the terminal.
LOG_DIR="$SCRIPT_DIR/logs/vllm"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${MODEL_NAME}_$(date +%Y%m%d_%H%M%S).log"
ln -sf "$(basename "$LOG_FILE")" "$LOG_DIR/${MODEL_NAME}_latest.log"
exec > >(tee -a "$LOG_FILE") 2>&1

# ── Launch ──────────────────────────────────────────────────────────────
echo "━━━ vLLM server: $MODEL_NAME ━━━"
echo "Config:  $CONFIG_FILE"
echo "Model:   $MODEL_PATH"
echo "Served:  $SERVED_NAME"
echo "Port:    $PORT"
echo "TP size: $TP_SIZE"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] && echo "GPUs:    $CUDA_VISIBLE_DEVICES"
echo "Command: ${CMD[*]}"
echo "Log:     $LOG_FILE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

exec "${CMD[@]}"
