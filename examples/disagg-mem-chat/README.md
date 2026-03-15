# Computer Architecture Project 9 — Track C: Disaggregated Memory Chat
### Modified LLM Inference Pipeline with Explicit Memory Placement

Built on [llama.cpp](https://github.com/ggml-org/llama.cpp).

---

## What This Is

A modified LLM chat tool that **instruments every inference step** and prints a real-time memory-hierarchy breakdown after each response. It simulates a 4-tier disaggregated memory system and reports:

- Where model weights and KV-cache are placed (SRAM / HBM / DRAM / FAR_MEM)
- Bytes moved per token (weight reads, KV reads, KV writes)
- Bandwidth-estimated decode latency and which tier is the bottleneck
- Arithmetic intensity vs. roofline ridge point (memory-bound vs compute-bound)
- KV-cache tier migration events and remote-memory access counts

---

## Files

```
examples/disagg-mem-chat/
├── disagg_mem_chat.cpp     # Modified inference pipeline (main program)
├── disagg_mem_sim.h        # 4-tier memory simulator (header-only)
├── CMakeLists.txt          # Build target: llama-disagg-mem-chat
├── run_pipeline.sh         # Automated multi-scenario test pipeline
├── pipeline_output/        # Output from all test scenarios (auto-generated)
├── report.md               # Full technical report with measured results
└── README.md               # This file

models/
└── smollm2-135m-instruct-q4_k_m.gguf   # 103.67 MB test model
```

---

## Build

From the `llama.cpp` root directory:

```bash
# Configure (first time only)
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build the disagg-mem-chat target
cmake --build build --target llama-disagg-mem-chat -j$(nproc)
```

The binary lands at `build/bin/llama-disagg-mem-chat`.

---

## Run

### Interactive chat (recommended)
```bash
./build/bin/llama-disagg-mem-chat \
    -m models/smollm2-135m-instruct-q4_k_m.gguf \
    -c 2048
```

Type a message and press Enter. After the model responds, the memory analysis table is printed automatically. Type an empty line to quit.

### Options

| Flag  | Default | Description                             |
|-------|---------|-----------------------------------------|
| `-m`  | (required) | Path to `.gguf` model file           |
| `-c`  | 2048    | Context size in tokens                  |
| `-ngl`| 99      | GPU layers to offload (0 = CPU only)    |

### Piped / scripted input
```bash
printf "What is a KV-cache?\n\n" | \
    ./build/bin/llama-disagg-mem-chat -m models/smollm2-135m-instruct-q4_k_m.gguf -ngl 0
```

---

## Run the Automated Pipeline

The pipeline runs 5 scenarios with increasing context length, captures all output, and prints a summary table.

```bash
cd examples/disagg-mem-chat

# CPU only (default)
bash run_pipeline.sh

# Custom model and output directory
bash run_pipeline.sh /path/to/model.gguf ./my_output

# Full GPU offload on H100 (ngl=99) with 4096 context
bash run_pipeline.sh /path/to/model.gguf ./my_output 99 4096
```

**Pipeline parameters (positional):**

| Position | Default | Description                                      |
|----------|---------|--------------------------------------------------|
| 1        | `../../models/smollm2-135m-instruct-q4_k_m.gguf` | Path to `.gguf` model file |
| 2        | `./pipeline_output` | Output directory for scenario files    |
| 3        | `0`     | GPU layers to offload (`-ngl`); set to `99` for full GPU |
| 4        | `2048`  | Context size in tokens (`-c`)                    |

Output files appear in `pipeline_output/`:
- `01_short_single_turn.txt` — baseline, KV-cache in SRAM
- `02_medium_single_turn.txt` — full-context, KV migrates to HBM
- `03_multi_turn_conversation.txt` — 4 turns, KV-cache growth tracked
- `04_long_prompt_prefill.txt` — large prefill batch
- `05_context_growth_stress.txt` — 7 turns stress test
- `summary.txt` — key metrics extracted from all scenarios

---

## Sample Output

```
┌─────────────────────────────────────────────────────────────┐
│         TRACK C · DISAGGREGATED MEMORY  (Turn  1)          │
├──────────────────────────┬──────────────────────────────────┤
│ MEMORY PLACEMENT         │ BANDWIDTH / LATENCY              │
├──────────────────────────┼──────────────────────────────────┤
│ Weights  103.67 MB       │ Tier 1 [HBM    ]                  │
│                          │   BW: 3.35 TB/s   lat:   10 ns   │
│ KV-cache 2.88 MB         │ Tier 0 [SRAM   ]                  │
│ (125 tokens cached)      │   BW: 10.00 TB/s  lat:    1 ns   │
│ Tier fill:  68.7%        │   Capacity: 4.19 MB               │
├──────────────────────────┴──────────────────────────────────┤
│ DATA MOVEMENT  (this turn: prefill 17 + decode 108 tokens)  │
│   Prefill weight reads : 103.67 MB  (once for prompt)       │
│   Decode  weight reads : 11.20 GB   (108 × weights)         │
│   KV reads (attention) : 181.05 MB  from Tier 0 [SRAM]      │
│   KV writes (new tok)  : 2.88 MB    to   Tier 0 [SRAM]      │
│   Turn total bytes     : 11.48 GB                           │
│   Bytes / gen token    : 105.37 MB                          │
├─────────────────────────────────────────────────────────────┤
│ LATENCY MODEL  (per last decode step)                       │
│   Weight BW time :    0.031 ms  ← Tier 1 [HBM] @ 3.35 TB/s │
│   KV read  time  :    0.000 ms  ← Tier 0 [SRAM] @ 10 TB/s  │
│   Bottleneck     : Weight reads                             │
├─────────────────────────────────────────────────────────────┤
│ ROOFLINE  (arithmetic intensity)                            │
│   FLOPs/token : ~269.03 MB  (2 × 134M params)              │
│   Bytes/token : 105.37 MB                                   │
│   AI          :   2.553 FLOP/byte  →  MEMORY-BOUND ✗       │
│   Ridge point :     295 FLOP/byte  (989 TFLOPS / 3.35 TB/s)│
├─────────────────────────────────────────────────────────────┤
│ SESSION TOTALS                                              │
│   Tokens generated : 108                                    │
│   Total bytes moved: 11.48 GB                               │
│   Tier migrations  : 0   (KV-cache moved between tiers)    │
│   Remote accesses  : 0   (Tier≥2: DRAM or FAR_MEM)         │
│   Actual gen speed : 405.2 tok/s                            │
└─────────────────────────────────────────────────────────────┘
```

---

## Memory Hierarchy (Simulated)

| Tier | Name     | Bandwidth   | Latency  | Capacity | Real-world Basis    |
|------|----------|-------------|----------|----------|---------------------|
| 0    | SRAM     | 10.00 TB/s  | 1 ns     | 4 MB     | On-chip SRAM        |
| 1    | HBM      | 3.35 TB/s   | 10 ns    | 32 GB    | HBM3 (A100/H100)    |
| 2    | DRAM     | 68 GB/s     | 80 ns    | 64 GB    | DDR5-5600           |
| 3    | FAR_MEM  | 25 GB/s     | 500 ns   | 512 GB   | CXL 3.0 memory pool |

**Placement policy:**
- **Model weights** → placed in the fastest tier large enough to hold them (always Tier 1 HBM for this model)
- **KV-cache** → placed in the fastest tier it currently fits in; migrates to a slower tier as context grows
- **KV-cache eviction point** ≈ 182 tokens (when 4 MB SRAM is exceeded)

---

## Key Findings (from report.md)

1. **Always memory-bound.** Arithmetic intensity ≈ 2 FLOP/byte, which is ~116× below the H100 roofline ridge point (295 FLOP/byte). Compute units are always stalling on memory.

2. **Weights dominate traffic.** Weight reads (103.67 MB/step) are 97–99% of all bytes moved per token. KV reads are secondary.

3. **Memory energy >> compute energy.** 418 µJ/token from memory vs. 0.54 µJ/token from MACs — a **775× ratio**.

4. **Generation speed falls with context.** From 405 tok/s at T=125 to 241 tok/s at T=1950 — a 40% drop driven by growing KV read volume.

5. **Disaggregation costs 40× latency per step** if KV-cache is evicted to FAR_MEM (25 GB/s) from HBM (3.35 TB/s). Current CXL bandwidth is 134× below HBM.

For the full analysis, see **[report.md](report.md)**.

---

## Model Download

The test model was downloaded from HuggingFace:

```bash
curl -L -o models/smollm2-135m-instruct-q4_k_m.gguf \
  "https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q4_K_M.gguf"
```

Any other `.gguf` model can be used with the `-m` flag.
