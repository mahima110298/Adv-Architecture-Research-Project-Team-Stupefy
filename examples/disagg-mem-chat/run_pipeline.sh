#!/usr/bin/env bash
# run_pipeline.sh
# COA Project — Track C: Disaggregated Memory Inference Pipeline
#
# Feeds a series of prompts to llama-disagg-mem-chat and captures the
# memory-hierarchy statistics for analysis and reporting.
#
# Usage:
#   ./run_pipeline.sh [model_path] [output_dir] [ngl] [ctx]
#
# Defaults:
#   model_path = ../../models/smollm2-135m-instruct-q4_k_m.gguf
#   output_dir = ./pipeline_output
#   ngl        = 0   (GPU layers to offload; set to 99 for full GPU offload)
#   ctx        = 2048

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BINARY="$SCRIPT_DIR/../../build/bin/llama-disagg-mem-chat"
MODEL="${1:-$SCRIPT_DIR/../../models/smollm2-135m-instruct-q4_k_m.gguf}"
OUT_DIR="${2:-$SCRIPT_DIR/pipeline_output}"
NGL="${3:-0}"
CTX="${4:-2048}"

mkdir -p "$OUT_DIR"

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
echo "  COA Track C — Disaggregated Memory Inference Pipeline"
echo "============================================================"
echo "  Binary : $BINARY"
echo "  Model  : $MODEL"
echo "  Output : $OUT_DIR"
echo "  NGL    : $NGL"
echo "  CTX    : $CTX"
echo "============================================================"
echo ""

# ── Helper: run one scenario and save output ──────────────────────────────────
run_scenario() {
    local name="$1"
    local prompts="$2"   # newline-separated, ends with blank line
    local outfile="$OUT_DIR/${name}.txt"

    echo "  [RUN] $name ..."
    # Strip ANSI color codes for the saved file
    printf "%s\n\n" "$prompts" \
        | "$BINARY" -m "$MODEL" -c $CTX -ngl $NGL 2>/dev/null \
        | sed 's/\x1b\[[0-9;]*m//g' \
        > "$outfile"
    echo "        saved → $outfile"
}

# ── Scenario 1: Short single-turn (baseline, small KV) ───────────────────────
run_scenario "01_short_single_turn" \
"What is 2 plus 2?"

# ── Scenario 2: Medium single-turn (more generated tokens) ───────────────────
run_scenario "02_medium_single_turn" \
"Explain what a transformer neural network is in detail."

# ── Scenario 3: Multi-turn conversation (KV-cache accumulates) ───────────────
run_scenario "03_multi_turn_conversation" \
"Hello, my name is Alice. Tell me about memory in computers.
What is the difference between DRAM and SRAM?
How does a CPU cache work?
What is the memory wall problem?"

# ── Scenario 4: Long single prompt (large prefill batch) ─────────────────────
run_scenario "04_long_prompt_prefill" \
"Please write a detailed explanation covering the following topics in order: (1) what is bandwidth in computer memory systems and why it matters, (2) what is the difference between on-chip SRAM and off-chip DRAM, (3) how does a KV-cache work in a large language model, (4) what is arithmetic intensity and what does it mean to be memory-bound versus compute-bound, and (5) what is disaggregated memory and how could it help large language model inference."

# ── Scenario 5: Repeated turns to grow KV-cache and trigger tier migration ───
run_scenario "05_context_growth_stress" \
"What is artificial intelligence?
Describe machine learning in simple terms.
What are neural networks and how do they learn?
What is backpropagation?
What is the difference between training and inference?
How does attention work in transformers?
What is a KV-cache?"

echo ""
echo "============================================================"
echo "  All scenarios complete. Extracting summary metrics..."
echo "============================================================"
echo ""

# ── Extract key metrics from each output file ─────────────────────────────────
SUMMARY="$OUT_DIR/summary.txt"
echo "COA Track C — Pipeline Summary" > "$SUMMARY"
echo "Generated: $(date)"             >> "$SUMMARY"
echo "Model: $(basename "$MODEL")"    >> "$SUMMARY"
echo ""                               >> "$SUMMARY"
printf "%-35s %12s %14s %14s %12s %10s %8s\n" \
    "Scenario" "KV-cache" "Bytes/token" "TotalBytes" "GenTok/s" "Migrations" "RemoteAcc" >> "$SUMMARY"
printf "%-35s %12s %14s %14s %12s %10s %8s\n" \
    "-----------------------------------" "----------" "------------" "------------" "----------" "----------" "---------" >> "$SUMMARY"

for f in "$OUT_DIR"/0*.txt; do
    name="$(basename "$f" .txt)"
    # Strip box-drawing │ chars before parsing so $NF is always the real last field
    kv_tok=$(grep -o 'KV-cache [0-9.]* [KMGT]B' "$f" | tail -1 | awk '{print $2$3}' || echo "?")
    bpt=$(grep 'Bytes / gen token' "$f" | tail -1 | sed 's/│//g' | awk '{print $(NF-1), $NF}' || echo "?")
    total=$(grep 'Total bytes moved' "$f" | tail -1 | sed 's/│//g' | awk '{print $(NF-1), $NF}' || echo "?")
    speed=$(grep 'Actual gen speed' "$f" | tail -1 | sed 's/│//g' | awk '{print $(NF-1), $NF}' || echo "?")
    migrations=$(grep 'Tier migrations' "$f" | tail -1 | grep -o '[0-9]*' | head -1 || echo "0")
    remote=$(grep 'Remote accesses' "$f" | tail -1 | grep -o '[0-9]*' | head -1 || echo "0")
    printf "%-35s %12s %14s %14s %12s %10s %8s\n" \
        "$name" "$kv_tok" "$bpt" "$total" "$speed" "$migrations" "$remote" >> "$SUMMARY"
done

echo "" >> "$SUMMARY"
echo "--- ROOFLINE (from last turn of last scenario) ---" >> "$SUMMARY"
last_file=$(ls "$OUT_DIR"/0*.txt | tail -1)
# Extract only the last ROOFLINE block (one per turn; we want the final state)
grep -A5 'ROOFLINE' "$last_file" | grep -v '^--$' | sed 's/│//g' | tail -6 >> "$SUMMARY" || true
echo "" >> "$SUMMARY"
echo "--- BOTTLENECK (from last scenario) ---" >> "$SUMMARY"
grep 'Bottleneck' "$last_file" | tail -1 | sed 's/│//g' >> "$SUMMARY" || true

cat "$SUMMARY"

echo ""
echo "============================================================"
echo "  Done. All output in: $OUT_DIR"
echo "============================================================"
