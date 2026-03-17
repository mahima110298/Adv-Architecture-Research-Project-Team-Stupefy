#!/usr/bin/env bash
# run_a100_tier_scenarios.sh
# COA Project — Track C: A100 KV-Cache Tier Migration Pipeline
#
# Runs 3 targeted scenarios on A100 to capture KV-cache in each memory tier:
#   Scenario 1 — KV in HBM     : ~15K token prompt, ctx=25000  → KV ≈ 12 GB  (HBM, <32 GB)
#   Scenario 2 — KV in DRAM    : ~43K token prompt, ctx=60000  → KV ≈ 34 GB  (DRAM, 32–64 GB)
#   Scenario 3 — KV in FAR_MEM : ~85K token prompt, ctx=100000 → KV ≈ 70 GB  (FAR_MEM, >64 GB)
#
# KV bytes/token for 13B (40 layers, 40 KV heads, 128 head_dim, fp16):
#   40 × 2 × 40 × 128 × 2 = 819,200 bytes ≈ 0.8 MB/token
#   HBM  threshold : 32 GB / 0.8 MB = ~41,943 tokens
#   DRAM threshold : 64 GB / 0.8 MB = ~83,886 tokens
#
# Usage:
#   bash run_a100_tier_scenarios.sh <model_path> [output_dir] [ngl]
#
# Example (A100, full GPU offload):
#   bash run_a100_tier_scenarios.sh /content/llama-2-13b-q4_k_m.gguf ./pipeline_output 99

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BINARY="$SCRIPT_DIR/../../build/bin/llama-disagg-mem-chat"
MODEL="${1:?Usage: $0 <model_path> [output_dir] [ngl]}"
OUT_DIR="${2:-$SCRIPT_DIR/pipeline_output}"
NGL="${3:-99}"

mkdir -p "$OUT_DIR"

# ── Validate binary and model ─────────────────────────────────────────────────
if [[ ! -f "$BINARY" ]]; then
    echo "ERROR: binary not found at $BINARY"
    echo "       Run: cmake --build ../../build --target llama-disagg-mem-chat"
    exit 1
fi
if [[ ! -f "$MODEL" ]]; then
    echo "ERROR: model not found at $MODEL"
    exit 1
fi

echo "============================================================"
echo "  COA Track C — A100 KV-Cache Tier Migration Pipeline"
echo "============================================================"
echo "  Binary : $BINARY"
echo "  Model  : $MODEL"
echo "  Output : $OUT_DIR"
echo "  NGL    : $NGL"
echo "============================================================"
echo ""
echo "  KV bytes/token (13B): ~0.8 MB"
echo "  HBM  → DRAM    threshold: ~41,943 tokens (32 GB)"
echo "  DRAM → FAR_MEM threshold: ~83,886 tokens (64 GB)"
echo "============================================================"
echo ""

# ── Helper: generate a repeated prompt of ~N tokens ──────────────────────────
# Each repetition of the seed sentence is ~28 tokens (conservative estimate).
# We subtract a 500-token safety margin from the target to avoid exceeding ctx.
make_prompt() {
    local target_tokens="$1"
    local reps=$(( (target_tokens - 500) / 28 + 1 ))
    python3 -c "
sentence = 'The history of computer architecture involves many important developments in processor design, memory systems, and instruction set architecture. '
print(sentence * $reps)
print()
"
}

# ── Helper: run one scenario ──────────────────────────────────────────────────
run_scenario() {
    local label="$1"       # e.g. "01_kv_in_hbm"
    local ctx="$2"         # context window size
    local prompt_tokens="$3"  # target prompt length in tokens
    local outfile="$OUT_DIR/${label}.txt"

    echo "  [RUN] $label  (ctx=$ctx, prompt≈${prompt_tokens} tokens) ..."
    make_prompt "$prompt_tokens" \
        | "$BINARY" -m "$MODEL" -c "$ctx" -ngl "$NGL" 2>&1 \
        | sed 's/\x1b\[[0-9;]*m//g' \
        > "$outfile"
    echo "        saved → $outfile"
}

# ── Scenario 1: KV-cache in HBM ──────────────────────────────────────────────
# Target: ~15,000 token prefill → KV ≈ 12 GB → fits in HBM (cap=32 GB)
# ctx=25000 gives 10,000 token decode headroom after prefill
run_scenario "01_kv_in_hbm" 25000 15000

# ── Scenario 2: KV-cache in DRAM ─────────────────────────────────────────────
# Target: ~43,000 token prefill → KV ≈ 34 GB → exceeds HBM (32 GB), lands in DRAM
# ctx=60000 gives ~17,000 token decode headroom after prefill
run_scenario "02_kv_in_dram" 60000 43000

# ── Scenario 3: KV-cache in FAR_MEM (CXL) ────────────────────────────────────
# Target: ~85,000 token prefill → KV ≈ 70 GB → exceeds DRAM (64 GB), lands in FAR_MEM
# ctx=100000 gives ~15,000 token decode headroom after prefill
run_scenario "03_kv_in_far_mem" 100000 85000

echo ""
echo "============================================================"
echo "  All scenarios complete. Extracting summary metrics..."
echo "============================================================"
echo ""

# ── Summary table ─────────────────────────────────────────────────────────────
SUMMARY="$OUT_DIR/a100_tier_summary.txt"
{
echo "COA Track C — A100 KV-Cache Tier Migration Summary"
echo "Generated : $(date)"
echo "Model     : $(basename "$MODEL")"
echo ""
printf "%-22s %10s %8s %14s %14s %12s %10s %10s\n" \
    "Scenario" "KV-Tier" "KV-Size" "Bytes/token" "TotalBytes" "GenTok/s" "Migrations" "RemoteAcc"
printf "%-22s %10s %8s %14s %14s %12s %10s %10s\n" \
    "----------------------" "----------" "--------" "--------------" "--------------" "------------" "----------" "----------"

for f in "$OUT_DIR"/0*.txt; do
    [[ -f "$f" ]] || continue
    name="$(basename "$f" .txt)"

    kv_size=$(grep -o 'KV-cache [0-9.]* [KMGT]B' "$f" | tail -1 | awk '{print $2$3}' 2>/dev/null || echo "?")
    kv_tier=$(grep 'KV-cache' "$f" | grep 'Tier [0-9]' | tail -1 | grep -o 'Tier [0-9] \[[A-Z_]*\]' | tail -1 || echo "?")
    bpt=$(grep 'Bytes / gen token' "$f" | tail -1 | sed 's/│//g' | awk '{print $(NF-1), $NF}' 2>/dev/null || echo "?")
    total=$(grep 'Total bytes moved' "$f" | tail -1 | sed 's/│//g' | awk '{print $(NF-1), $NF}' 2>/dev/null || echo "?")
    speed=$(grep 'Actual gen speed' "$f" | tail -1 | sed 's/│//g' | awk '{print $(NF-1), $NF}' 2>/dev/null || echo "?")
    migrations=$(grep 'Tier migrations' "$f" | tail -1 | grep -o '[0-9]*' | head -1 2>/dev/null || echo "0")
    remote=$(grep 'Remote accesses' "$f" | tail -1 | grep -o '[0-9]*' | head -1 2>/dev/null || echo "0")

    printf "%-22s %10s %8s %14s %14s %12s %10s %10s\n" \
        "$name" "$kv_tier" "$kv_size" "$bpt" "$total" "$speed" "$migrations" "$remote"
done

echo ""
echo "--- LATENCY BREAKDOWN PER SCENARIO (last decode step) ---"
echo ""
for f in "$OUT_DIR"/0*.txt; do
    [[ -f "$f" ]] || continue
    name="$(basename "$f" .txt)"
    echo "[$name]"
    grep -A3 'LATENCY MODEL' "$f" | tail -1 | grep -v '^--$' | sed 's/│//g' | sed 's/^[[:space:]]*/  /' || true
    grep 'Weight BW time\|KV read  time\|Bottleneck' "$f" | tail -3 | sed 's/│//g' | sed 's/^[[:space:]]*/  /' || true
    echo ""
done

echo "--- SPEED COMPARISON ---"
echo ""
echo "  Expected impact of KV-cache tier:"
echo "    HBM     (3.35 TB/s) → baseline speed"
echo "    DRAM    (68  GB/s)  → ~49x slower KV reads"
echo "    FAR_MEM (25  GB/s)  → ~134x slower KV reads"
echo ""
} > "$SUMMARY"

cat "$SUMMARY"

echo ""
echo "============================================================"
echo "  Done. Full outputs:"
for f in "$OUT_DIR"/0*.txt; do
    [[ -f "$f" ]] && echo "    $f"
done
echo "  Summary: $SUMMARY"
echo "============================================================"
