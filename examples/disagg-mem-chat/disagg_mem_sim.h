// disagg_mem_sim.h
// Track C: Disaggregated Memory Simulation for LLM Inference
// COA Project — Memory-Centric Architecture
//
// Models a 4-tier memory hierarchy:
//   Tier 0 — On-chip SRAM      :  4 MB   |  10.0 TB/s |   1 ns  (hot scratch/KV)
//   Tier 1 — HBM/Stacked DRAM  : 32 GB   |   3.35 TB/s|  10 ns  (weights + warm KV)
//   Tier 2 — Off-chip DRAM     : 64 GB   |  68  GB/s  |  80 ns  (overflow KV)
//   Tier 3 — Far/CXL Memory    : 512 GB  |  25  GB/s  | 500 ns  (disaggregated, cold KV)
//
// Placement policy:
//   Weights  — placed in the fastest tier they fit in (Tier 1 or higher)
//   KV-cache — placed in the fastest tier it fits in, migrates out as context grows

#pragma once

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <string>

// ── Memory Tier Definitions ───────────────────────────────────────────────────

enum class MemTier : int {
    SRAM    = 0,
    HBM     = 1,
    DRAM    = 2,
    FAR_MEM = 3,
};

struct TierConfig {
    const char * name;
    double   bandwidth_TBs;    // TB/s (terabytes per second)
    double   latency_ns;       // nanoseconds
    int64_t  capacity_bytes;   // bytes
};

static const TierConfig TIER_CONFIGS[] = {
    // name       BW(TB/s)  lat(ns)  capacity
    { "SRAM",     10.000,     1.0,    4LL * 1024 * 1024              }, // 4 MB
    { "HBM",       3.350,    10.0,   32LL * 1024 * 1024 * 1024       }, // 32 GB
    { "DRAM",      0.068,    80.0,   64LL * 1024 * 1024 * 1024       }, // 64 GB
    { "FAR_MEM",   0.025,   500.0,  512LL * 1024 * 1024 * 1024       }, // 512 GB
};

static const int N_TIERS = 4;

// ── Formatting helpers ────────────────────────────────────────────────────────

static std::string fmt_bytes(double b) {
    char buf[64];
    if      (b >= 1e12) snprintf(buf, sizeof(buf), "%.2f TB",  b / 1e12);
    else if (b >= 1e9)  snprintf(buf, sizeof(buf), "%.2f GB",  b / 1e9);
    else if (b >= 1e6)  snprintf(buf, sizeof(buf), "%.2f MB",  b / 1e6);
    else if (b >= 1e3)  snprintf(buf, sizeof(buf), "%.2f KB",  b / 1e3);
    else                snprintf(buf, sizeof(buf), "%.0f B",   b);
    return std::string(buf);
}

static std::string fmt_bw(double tb_per_s) {
    char buf[32];
    if (tb_per_s >= 1.0) snprintf(buf, sizeof(buf), "%.2f TB/s", tb_per_s);
    else                  snprintf(buf, sizeof(buf), "%.1f GB/s", tb_per_s * 1000.0);
    return std::string(buf);
}

// ── DisaggMemSim ─────────────────────────────────────────────────────────────

class DisaggMemSim {
public:
    // ── Model parameters (set once at init) ─────────────────────────────────
    int64_t model_weight_bytes = 0;
    int32_t n_layers           = 0;
    int32_t n_kv_heads         = 0;
    int32_t n_heads            = 0;
    int32_t n_embd             = 0;
    int32_t head_dim           = 0;
    int64_t n_params           = 0;
    int32_t kv_elem_bytes      = 2; // fp16 by default

    // ── Session accumulators ──────────────────────────────────────────────────
    int64_t session_tokens_generated = 0;
    int64_t session_bytes_moved      = 0;
    int32_t session_tier_migrations  = 0;
    int32_t session_remote_accesses  = 0; // accesses to Tier >= 2 (DRAM/FAR)

    // ── Per-turn accumulators (reset each turn) ───────────────────────────────
    int64_t turn_tokens_generated    = 0;
    int64_t turn_prefill_tokens      = 0;
    int64_t turn_weight_bytes_decode = 0; // weights read during decode steps
    int64_t turn_weight_bytes_prefill= 0; // weights read during prefill (once)
    int64_t turn_kv_read_bytes       = 0;
    int64_t turn_kv_write_bytes      = 0;

    // ── Internal state ────────────────────────────────────────────────────────
    MemTier prev_kv_tier = MemTier::SRAM;
    int32_t turn_index   = 0;

    // ─────────────────────────────────────────────────────────────────────────

    void init(int64_t weight_bytes, int32_t layers, int32_t kv_heads,
              int32_t heads, int32_t embd, int64_t params) {
        model_weight_bytes = weight_bytes;
        n_layers           = layers;
        n_kv_heads         = kv_heads;
        n_heads            = heads;
        n_embd             = embd;
        head_dim           = (heads > 0) ? (embd / heads) : 64;
        n_params           = params;
        prev_kv_tier       = MemTier::SRAM;
    }

    void reset_turn() {
        turn_tokens_generated     = 0;
        turn_prefill_tokens       = 0;
        turn_weight_bytes_decode  = 0;
        turn_weight_bytes_prefill = 0;
        turn_kv_read_bytes        = 0;
        turn_kv_write_bytes       = 0;
    }

    // Called once for the prefill (prompt) batch.
    // kv_tokens_before: KV-cache size before the prompt was added.
    // n_prompt: number of prompt tokens being processed.
    void record_prefill(int32_t n_prompt, int32_t kv_tokens_after) {
        // Weights are read once (prefill processes all tokens in one pass)
        int64_t w_bytes = model_weight_bytes;

        // KV writes: one K+V pair per prompt token per layer
        int64_t kv_write = (int64_t)n_layers * n_prompt
                         * 2 * n_kv_heads * head_dim * kv_elem_bytes;

        // KV reads: during prefill, causal attention — average n_prompt/2 prev tokens
        // Simplified: reads grow linearly through the prompt (triangular pattern)
        int64_t kv_read = (int64_t)n_layers * (n_prompt * (int64_t)(n_prompt - 1) / 2)
                        * 2 * n_kv_heads * head_dim * kv_elem_bytes;

        turn_prefill_tokens        += n_prompt;
        turn_weight_bytes_prefill  += w_bytes;
        turn_kv_write_bytes        += kv_write;
        turn_kv_read_bytes         += kv_read;

        int64_t total_moved = w_bytes + kv_write + kv_read;
        session_bytes_moved += total_moved;

        _check_migration(kv_tokens_after);
    }

    // Called once per generated token (decode step, batch=1).
    // kv_tokens: total tokens in KV-cache AFTER this token is added.
    void record_decode_step(int32_t kv_tokens) {
        // Weights: read all once per token
        int64_t w_bytes = model_weight_bytes;

        // KV write: new K+V for this single token
        int64_t kv_write = (int64_t)n_layers * 2 * n_kv_heads * head_dim * kv_elem_bytes;

        // KV read: all previous K+V for attention across all layers
        int64_t kv_read  = (int64_t)n_layers * kv_tokens
                         * 2 * n_kv_heads * head_dim * kv_elem_bytes;

        turn_weight_bytes_decode += w_bytes;
        turn_kv_write_bytes      += kv_write;
        turn_kv_read_bytes       += kv_read;
        turn_tokens_generated++;

        session_tokens_generated++;
        session_bytes_moved += w_bytes + kv_read + kv_write;

        _check_migration(kv_tokens);
        if ((int)prev_kv_tier >= 2) {
            session_remote_accesses++;
        }
    }

    // ── Queries ───────────────────────────────────────────────────────────────

    int64_t kv_bytes_total(int32_t kv_tokens) const {
        return (int64_t)kv_tokens * n_layers * 2 * n_kv_heads * head_dim * kv_elem_bytes;
    }

    MemTier weight_tier() const {
        for (int i = 1; i < N_TIERS; i++) { // weights can't fit in SRAM
            if (model_weight_bytes <= TIER_CONFIGS[i].capacity_bytes)
                return static_cast<MemTier>(i);
        }
        return MemTier::FAR_MEM;
    }

    MemTier kv_tier(int64_t kv_bytes) const {
        for (int i = 0; i < N_TIERS; i++) {
            if (kv_bytes <= TIER_CONFIGS[i].capacity_bytes)
                return static_cast<MemTier>(i);
        }
        return MemTier::FAR_MEM;
    }

    // Estimated bandwidth-limited time to read 'bytes' from 'tier' (in ms)
    double latency_bw_ms(int64_t bytes, MemTier tier) const {
        double bw = TIER_CONFIGS[(int)tier].bandwidth_TBs * 1e12; // bytes/s
        return (bytes / bw) * 1000.0;
    }

    // Arithmetic intensity for the decode portion of this turn (FLOP/byte)
    double arithmetic_intensity() const {
        if (turn_tokens_generated == 0) return 0.0;
        double flops_per_token = 2.0 * (double)n_params;
        double bytes_per_token = (double)(turn_weight_bytes_decode
                                        + turn_kv_read_bytes
                                        + turn_kv_write_bytes)
                                / turn_tokens_generated;
        return (bytes_per_token > 0) ? (flops_per_token / bytes_per_token) : 0.0;
    }

    // ── Print ─────────────────────────────────────────────────────────────────

    void print_turn_stats(int32_t kv_tokens_after, double gen_time_ms) {
        turn_index++;

        int64_t kv_total = kv_bytes_total(kv_tokens_after);
        MemTier w_tier   = weight_tier();
        MemTier kv_t     = kv_tier(kv_total);

        const TierConfig & w_cfg  = TIER_CONFIGS[(int)w_tier];
        const TierConfig & kv_cfg = TIER_CONFIGS[(int)kv_t];

        // Per-token bytes (decode phase only, which is the bottleneck)
        double bytes_per_tok = 0;
        if (turn_tokens_generated > 0) {
            bytes_per_tok = (double)(turn_weight_bytes_decode
                                   + turn_kv_read_bytes
                                   + turn_kv_write_bytes)
                           / turn_tokens_generated;
        }

        // Latency per last decode step (worst case: longest KV read)
        int64_t last_kv_read = (int64_t)n_layers * kv_tokens_after
                             * 2 * n_kv_heads * head_dim * kv_elem_bytes;
        double w_lat   = latency_bw_ms(model_weight_bytes, w_tier);
        double kv_lat  = latency_bw_ms(last_kv_read, kv_t);

        double ai = arithmetic_intensity();
        // Ridge point on the roofline: peak_compute / peak_bw
        // Using H100-class HBM numbers as reference point
        const double peak_tflops = 989.0;   // H100 BF16 tensor cores (TFLOPS)
        const double ridge_pt    = (peak_tflops * 1e12) / (w_cfg.bandwidth_TBs * 1e12);

        double kv_fill_pct = 100.0 * (double)kv_total / kv_cfg.capacity_bytes;

        int64_t turn_total_bytes = turn_weight_bytes_prefill
                                 + turn_weight_bytes_decode
                                 + turn_kv_read_bytes
                                 + turn_kv_write_bytes;

        printf("\n\033[36m"); // cyan
        printf("┌─────────────────────────────────────────────────────────────┐\n");
        printf("│         TRACK C · DISAGGREGATED MEMORY  (Turn %2d)          │\n", turn_index);
        printf("├──────────────────────────┬──────────────────────────────────┤\n");
        printf("│ MEMORY PLACEMENT         │ BANDWIDTH / LATENCY              │\n");
        printf("├──────────────────────────┼──────────────────────────────────┤\n");
        printf("│ Weights  %-10s      │ Tier %d [%-7s]                  │\n",
               fmt_bytes(model_weight_bytes).c_str(), (int)w_tier, w_cfg.name);
        printf("│                          │   BW: %-10s  lat: %4.0f ns   │\n",
               fmt_bw(w_cfg.bandwidth_TBs).c_str(), w_cfg.latency_ns);
        printf("│ KV-cache %-10s      │ Tier %d [%-7s]                  │\n",
               fmt_bytes(kv_total).c_str(), (int)kv_t, kv_cfg.name);
        printf("│ (%d tokens cached)        │   BW: %-10s  lat: %4.0f ns   │\n",
               kv_tokens_after, fmt_bw(kv_cfg.bandwidth_TBs).c_str(), kv_cfg.latency_ns);
        printf("│ Tier fill: %5.1f%%         │   Capacity: %-16s   │\n",
               kv_fill_pct, fmt_bytes(kv_cfg.capacity_bytes).c_str());
        printf("├──────────────────────────┴──────────────────────────────────┤\n");
        printf("│ DATA MOVEMENT  (this turn: prefill %lld + decode %lld tokens)   │\n",
               (long long)turn_prefill_tokens, (long long)turn_tokens_generated);
        printf("│   Prefill weight reads : %-12s (once for prompt)   │\n",
               fmt_bytes(turn_weight_bytes_prefill).c_str());
        printf("│   Decode  weight reads : %-12s (%lld × weights)  │\n",
               fmt_bytes(turn_weight_bytes_decode).c_str(), (long long)turn_tokens_generated);
        printf("│   KV reads (attention) : %-12s from Tier %d [%s]  │\n",
               fmt_bytes(turn_kv_read_bytes).c_str(), (int)kv_t, kv_cfg.name);
        printf("│   KV writes (new tok)  : %-12s to   Tier %d [%s]  │\n",
               fmt_bytes(turn_kv_write_bytes).c_str(), (int)kv_t, kv_cfg.name);
        printf("│   Turn total bytes     : %-12s                      │\n",
               fmt_bytes(turn_total_bytes).c_str());
        printf("│   Bytes / gen token    : %-12s                      │\n",
               fmt_bytes(bytes_per_tok).c_str());
        printf("├─────────────────────────────────────────────────────────────┤\n");
        printf("│ LATENCY MODEL  (per last decode step)                       │\n");
        printf("│   Weight BW time : %8.3f ms  ← Tier %d [%s] @ %s   │\n",
               w_lat, (int)w_tier, w_cfg.name, fmt_bw(w_cfg.bandwidth_TBs).c_str());
        printf("│   KV read  time  : %8.3f ms  ← Tier %d [%s] @ %s    │\n",
               kv_lat, (int)kv_t, kv_cfg.name, fmt_bw(kv_cfg.bandwidth_TBs).c_str());
        printf("│   Bottleneck     : %-40s │\n",
               (w_lat >= kv_lat) ? "Weight reads  (model too large for faster tier)"
                                 : "KV-cache reads (context too long for faster tier)");
        printf("├─────────────────────────────────────────────────────────────┤\n");
        printf("│ ROOFLINE  (arithmetic intensity)                            │\n");
        printf("│   FLOPs/token : ~%-12s  (2 × %lld params)          │\n",
               fmt_bytes(2.0 * n_params).c_str(), (long long)n_params);
        printf("│   Bytes/token : %-12s                               │\n",
               fmt_bytes(bytes_per_tok).c_str());
        printf("│   AI          : %7.3f FLOP/byte  →  %-16s     │\n",
               ai, (ai < ridge_pt) ? "MEMORY-BOUND ✗" : "COMPUTE-BOUND ✓");
        printf("│   Ridge point : %7.0f FLOP/byte  (989 TFLOPS / %.2f TB/s)│\n",
               ridge_pt, w_cfg.bandwidth_TBs);
        printf("├─────────────────────────────────────────────────────────────┤\n");
        printf("│ SESSION TOTALS                                               │\n");
        printf("│   Tokens generated : %-6lld                                 │\n",
               (long long)session_tokens_generated);
        printf("│   Total bytes moved: %-12s                           │\n",
               fmt_bytes(session_bytes_moved).c_str());
        printf("│   Tier migrations  : %-3d  (KV-cache moved between tiers)  │\n",
               session_tier_migrations);
        printf("│   Remote accesses  : %-6d (Tier≥2: DRAM or FAR_MEM)     │\n",
               session_remote_accesses);
        if (gen_time_ms > 0 && turn_tokens_generated > 0) {
            printf("│   Actual gen speed : %5.1f tok/s                           │\n",
                   turn_tokens_generated * 1000.0 / gen_time_ms);
        }
        printf("└─────────────────────────────────────────────────────────────┘\n");
        printf("\033[0m"); // reset color
        fflush(stdout);
    }

private:
    void _check_migration(int32_t kv_tokens_after) {
        int64_t kv_total  = kv_bytes_total(kv_tokens_after);
        MemTier cur_tier  = kv_tier(kv_total);
        if (cur_tier != prev_kv_tier) {
            session_tier_migrations++;
            prev_kv_tier = cur_tier;
        }
    }
};
