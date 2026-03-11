// disagg_mem_chat.cpp
// COA Project — Track C: Disaggregated & Shared Memory Systems
//
// An interactive LLM chat tool that instruments every decode step and
// prints a detailed memory-hierarchy report after each response, showing:
//   - Where model weights and KV-cache live (SRAM / HBM / DRAM / FAR_MEM)
//   - Bytes moved per token (weights + KV reads + KV writes)
//   - Estimated bandwidth-latency breakdown per decode step
//   - Arithmetic intensity vs. roofline ridge point
//   - Tier migrations and remote-memory access counts

#include "llama.h"
#include "disagg_mem_sim.h"

#include <chrono>
#include <clocale>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

static void print_usage(int, char ** argv) {
    printf("\nusage:\n");
    printf("  %s -m model.gguf [-c context_size] [-ngl n_gpu_layers]\n\n", argv[0]);
    printf("options:\n");
    printf("  -m   <path>   path to .gguf model file (required)\n");
    printf("  -c   <int>    context size in tokens    (default: 2048)\n");
    printf("  -ngl <int>    GPU layers to offload     (default: 99)\n\n");
}

// ── ANSI helpers ──────────────────────────────────────────────────────────────
#define COL_GRN  "\033[32m"
#define COL_YEL  "\033[33m"
#define COL_RST  "\033[0m"

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    // ── Parse args ────────────────────────────────────────────────────────────
    std::string model_path;
    int ngl   = 99;
    int n_ctx = 2048;

    for (int i = 1; i < argc; i++) {
        try {
            if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
                model_path = argv[++i];
            } else if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) {
                n_ctx = std::stoi(argv[++i]);
            } else if (strcmp(argv[i], "-ngl") == 0 && i + 1 < argc) {
                ngl = std::stoi(argv[++i]);
            } else {
                print_usage(argc, argv);
                return 1;
            }
        } catch (std::exception & e) {
            fprintf(stderr, "error: %s\n", e.what());
            print_usage(argc, argv);
            return 1;
        }
    }
    if (model_path.empty()) { print_usage(argc, argv); return 1; }

    // ── Suppress non-error logs ───────────────────────────────────────────────
    llama_log_set([](enum ggml_log_level level, const char * text, void *) {
        if (level >= GGML_LOG_LEVEL_ERROR) fprintf(stderr, "%s", text);
    }, nullptr);

    // ── Load model ────────────────────────────────────────────────────────────
    ggml_backend_load_all();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) { fprintf(stderr, "error: failed to load model\n"); return 1; }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    // ── Init context ──────────────────────────────────────────────────────────
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx   = n_ctx;
    cp.n_batch = n_ctx;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "error: failed to create context\n"); return 1; }

    // ── Init sampler ──────────────────────────────────────────────────────────
    llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_min_p(0.05f, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.8f));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    // ── Collect model parameters for the memory simulator ─────────────────────
    int32_t n_layers   = llama_model_n_layer(model);
    int32_t n_kv_heads = llama_model_n_head_kv(model);
    int32_t n_heads    = llama_model_n_head(model);
    int32_t n_embd     = llama_model_n_embd(model);
    int64_t n_params   = (int64_t)llama_model_n_params(model);
    int64_t weight_bytes = (int64_t)llama_model_size(model);

    // ── Print model summary ───────────────────────────────────────────────────
    printf(COL_GRN);
    printf("┌─────────────────────────────────────────────────────────────┐\n");
    printf("│   COA Project · Track C: Disaggregated Memory Chat          │\n");
    printf("├─────────────────────────────────────────────────────────────┤\n");
    printf("│  Model parameters  : %-12s (%.1fB params)            │\n",
           fmt_bytes(n_params).c_str(), n_params / 1e9);
    printf("│  Weight footprint  : %-12s (on disk/in-memory)       │\n",
           fmt_bytes(weight_bytes).c_str());
    printf("│  Layers            : %-4d                                   │\n", n_layers);
    printf("│  KV heads          : %-4d  (heads: %d, embd: %d)            │\n",
           n_kv_heads, n_heads, n_embd);
    printf("│  Context size      : %-6d tokens                         │\n", n_ctx);
    int32_t head_dim = (n_heads > 0) ? n_embd / n_heads : 64;
    int64_t max_kv   = (int64_t)n_ctx * n_layers * 2 * n_kv_heads * head_dim * 2;
    printf("│  Max KV-cache      : %-12s (at full context, fp16)    │\n",
           fmt_bytes(max_kv).c_str());
    printf("├─────────────────────────────────────────────────────────────┤\n");
    printf("│  Memory Hierarchy (simulated):                              │\n");
    for (int i = 0; i < N_TIERS; i++) {
        printf("│    Tier %d [%-7s]: BW=%-10s  lat=%4.0f ns  cap=%-8s │\n",
               i, TIER_CONFIGS[i].name,
               fmt_bw(TIER_CONFIGS[i].bandwidth_TBs).c_str(),
               TIER_CONFIGS[i].latency_ns,
               fmt_bytes(TIER_CONFIGS[i].capacity_bytes).c_str());
    }
    printf("├─────────────────────────────────────────────────────────────┤\n");
    printf("│  Type a message and press Enter. Empty line to quit.        │\n");
    printf("└─────────────────────────────────────────────────────────────┘\n");
    printf(COL_RST);

    // ── Init memory simulator ─────────────────────────────────────────────────
    DisaggMemSim sim;
    sim.init(weight_bytes, n_layers, n_kv_heads, n_heads, n_embd, n_params);

    // ── Generate function with per-step memory tracking ───────────────────────
    auto generate = [&](const std::string & prompt) -> std::string {
        std::string response;
        sim.reset_turn();

        const bool is_first = llama_memory_seq_pos_max(llama_get_memory(ctx), 0) == -1;

        // Tokenize
        int n_prompt_tokens = -llama_tokenize(
            vocab, prompt.c_str(), prompt.size(), nullptr, 0, is_first, true);
        std::vector<llama_token> prompt_tokens(n_prompt_tokens);
        if (llama_tokenize(vocab, prompt.c_str(), prompt.size(),
                           prompt_tokens.data(), prompt_tokens.size(),
                           is_first, true) < 0) {
            GGML_ABORT("failed to tokenize prompt\n");
        }

        llama_batch batch     = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
        llama_token new_token = 0;
        bool   first_batch    = true;
        auto   gen_start      = std::chrono::steady_clock::now();

        while (true) {
            int n_ctx_cur  = llama_n_ctx(ctx);
            int n_ctx_used = llama_memory_seq_pos_max(llama_get_memory(ctx), 0) + 1;
            if (n_ctx_used + batch.n_tokens > n_ctx_cur) {
                printf("\n" COL_RST);
                fprintf(stderr, "context size exceeded\n");
                break;
            }

            int ret = llama_decode(ctx, batch);
            if (ret != 0) GGML_ABORT("decode failed, ret=%d\n", ret);

            // KV-cache size after this decode
            int kv_now = llama_memory_seq_pos_max(llama_get_memory(ctx), 0) + 1;

            if (first_batch) {
                // Prefill step
                sim.record_prefill(batch.n_tokens, kv_now);
                first_batch = false;
            } else {
                // Decode step (single token)
                sim.record_decode_step(kv_now);
            }

            new_token = llama_sampler_sample(smpl, ctx, -1);
            if (llama_vocab_is_eog(vocab, new_token)) break;

            char buf[256];
            int  n = llama_token_to_piece(vocab, new_token, buf, sizeof(buf), 0, true);
            if (n < 0) GGML_ABORT("failed to convert token\n");
            std::string piece(buf, n);
            printf("%s", piece.c_str());
            fflush(stdout);
            response += piece;

            batch = llama_batch_get_one(&new_token, 1);
        }

        auto gen_end    = std::chrono::steady_clock::now();
        double gen_ms   = std::chrono::duration<double, std::milli>(gen_end - gen_start).count();
        int    kv_final = llama_memory_seq_pos_max(llama_get_memory(ctx), 0) + 1;

        printf("\n" COL_RST);
        sim.print_turn_stats(kv_final, gen_ms);

        return response;
    };

    // ── Chat loop ─────────────────────────────────────────────────────────────
    std::vector<llama_chat_message> messages;
    std::vector<char> formatted(llama_n_ctx(ctx));
    int prev_len = 0;

    while (true) {
        printf(COL_GRN "> " COL_RST);
        std::string user;
        std::getline(std::cin, user);
        if (user.empty()) break;

        const char * tmpl = llama_model_chat_template(model, nullptr);
        messages.push_back({"user", strdup(user.c_str())});

        int new_len = llama_chat_apply_template(
            tmpl, messages.data(), messages.size(), true,
            formatted.data(), formatted.size());
        if (new_len > (int)formatted.size()) {
            formatted.resize(new_len);
            new_len = llama_chat_apply_template(
                tmpl, messages.data(), messages.size(), true,
                formatted.data(), formatted.size());
        }
        if (new_len < 0) { fprintf(stderr, "chat template error\n"); return 1; }

        std::string prompt(formatted.begin() + prev_len, formatted.begin() + new_len);

        printf(COL_YEL);
        std::string response = generate(prompt);
        printf(COL_RST "\n");

        messages.push_back({"assistant", strdup(response.c_str())});
        prev_len = llama_chat_apply_template(
            tmpl, messages.data(), messages.size(), false, nullptr, 0);
        if (prev_len < 0) { fprintf(stderr, "chat template error\n"); return 1; }
    }

    // ── Cleanup ───────────────────────────────────────────────────────────────
    for (auto & m : messages) free(const_cast<char *>(m.content));
    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
