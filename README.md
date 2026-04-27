# OllamaSi by Core-Fidelity

Apple Silicon optimizations for [Ollama](https://github.com/ollama/ollama), focused on unified-memory behavior, mlock policy, and zero-copy weight loading on macOS.

This is a fork of upstream Ollama with targeted changes to the Metal backend and model loader. The goal is to eliminate pathological memory-related regressions on Apple Silicon without breaking the experience on other platforms.

---

## What Changed

### Unified-memory weight loading (`host_ptr` / `buffer_from_host_ptr`)

On Apple Silicon, model weights can be mapped directly into GPU-visible memory via `mmap` + Metal `newBufferWithBytesNoCopy:` with `MTLResourceStorageModeShared`. This avoids the double-copy that the default Ollama path performs on Metal (one host-side copy in ggml buffers, then another GPU-side upload).

OllamaSi enables this zero-copy path by default on Metal when the backend supports `buffer_from_host_ptr`.

### mlock policy fix

Upstream Ollama uses a percentage-based post-mlock threshold (6.5% of total physical RAM) to decide whether to keep pages pinned after `mlock()` succeeds. On 8 GB Apple Silicon systems, this threshold is hit immediately by virtually every model that successfully `mlock`s (2–4B parameter range). The resulting behavior was:

1. `mlock()` succeeds.
2. The 6.5% check immediately fails (`~524 MB` required, but far less is free after the lock).
3. Pages are `munlock()`ed.
4. Decode and prefill regress because the model is no longer pinned and now suffers page-fault thrashing.

OllamaSi replaces this with a fixed absolute floor:

```go
const minHostPtrHeadroomBytes = 512 * 1024 * 1024 // 512 MB
```

This preserves three distinct runtime paths:

| Path | Trigger | Behavior |
|---|---|---|
| **Pinned** | `mlock` succeeds + ≥ 512 MB headroom remains | Pages stay pinned. Best decode/prefill stability. |
| **Unpinned zero-copy** | `mlock` fails with `ENOMEM` | `mmap` stays active, pages fault in lazily. Still zero-copy; load is fast. |
| **Copy fallback** | `OLLAMA_USE_HOST_PTR=0` or structural failure | Standard Ollama copy path. Safe, slower load. |

### Metal partial-offload `mmap` gating removed

Upstream unconditionally disables `mmap` for Metal partial offload. On Apple Silicon this forces a full memory copy even when the model is slightly too large for the GPU, causing severe swap pressure on 8 GB systems. OllamaSi restores the `mmap` + partial-offload path that `llama.cpp` already supports.

### Prefix KV cache (on-disk prefix persistence)

Immutable prefix tokens (`num_keep`, typically system prompts) are tracked separately in the causal KV cache and never shifted or evicted. After the first full prefix computation, the KV state is serialized to `~/.ollama/kvcache/` and restored on subsequent requests, skipping recompute.

Hybrid models (Qwen3-Next, etc.) store their recurrent state alongside causal KV data and produce unique cache keys so that attention and recurrent architectures never collide.

---

## Why This Was Needed

The motivating bug was a pathological interaction between `mlock`, unified memory, and percentage-based thresholds on small-memory Apple Silicon machines. A 3–4B model like `qwen3.5:4b` would:

- Load slowly due to copy-path weight duplication.
- Decode at ~6.9 tok/s instead of ~11.9 tok/s because pinned pages were immediately released.
- Prefill at ~47 tok/s instead of ~91 tok/s due to page-fault overhead.

For large models like `gemma4:e2b` (~6.67 GB), the problem was different: instead of falling back cleanly to the unpinned zero-copy path when `mlock` failed, the old code could enter a degenerate copy-path that regressed load time by 10–20x.

The fix is a single consistent memory policy: `mlock` when safe, zero-copy fallback when not, never silently degrade.

---

## Benchmarks

All numbers cold-start, no warmup, measured with the internal benchmark harness.

### `gemma4:e2b` — 6.67 GB (pathological load case)

| Metric | Baseline | OllamaSi | Note |
|---|---|---|---|
| Load | ~13 s+ | ~1.1 s | Copy-path regression eliminated |
| Decode | ~35–36 tok/s | ~35 tok/s | No regression; zero-copy preserved |
| TTFT | Severe | ~1.2–1.3 s | First-token latency recovered |

The 13-second baseline was caused by full weight duplication through the copy path when zero-copy should have been active. OllamaSi restores the fast path.

### `qwen3.5:4b` — recovery from always-unlock bug

| Metric | Baseline | OllamaSi | Delta |
|---|---|---|---|
| Load | 2,886 ms | 1,460 ms | –49 % |
| TTFT | 4,171 ms | 2,003 ms | –52 % |
| Prefill | 47.2 tok/s | 91.7 tok/s | +94 % |
| Decode | 6.9 tok/s | 11.9 tok/s | +72 % |

This is the clearest demonstration of the old mlock unlock-after-success pathology.

### Models observed with correct behavior under the 512 MB policy

Pinned correctly (small-to-medium models on 8 GB):

- `phi3-mini`
- `qwen3.5:2b`
- `medgemma1.5:4b`
- `translategemma:4b`
- `qwen3.5:4b`

Unpinned correctly via `ENOMEM` fallback (large models):

- `gemma4:e2b`

No model-specific gates or heuristics are used. The same policy applies to every model.

---

## Technical Explanation

### The memory model on Apple Silicon

Apple Silicon uses unified memory: the CPU and GPU share the same physical DRAM. Metal can create GPU-visible buffers directly from host `mmap` pages without a copy, via `newBufferWithBytesNoCopy:` with `MTLResourceStorageModeShared`. This means:

- **Zero-copy path**: `mmap` the model file once → Metal buffer wraps those pages → CPU layers read directly from the same pages.
- **Copy path**: `mmap` the file → allocate ggml buffers → copy tensor data into those buffers → upload to GPU buffers. This creates 2–3x memory pressure depending on overlap.

### The old mlock threshold

`mlock` pins pages in RAM, preventing the OS from reclaiming them. On memory-constrained systems, paging out model weights during inference causes the GPU to fault them back in from disk, destroying throughput. So pinning is valuable.

The old policy was:

```go
if availableAfterMlock < totalPhysicalMemory * 0.065 { munlock() }
```

On an 8 GB Mac, `0.065 * 8 GB = ~524 MB`. After a successful `mlock` of a 3–4 GB model, free memory is almost always below that threshold, so `munlock()` fires immediately. The model pays the `mlock` cost and receives none of the benefit.

### The fixed policy

```go
const minHostPtrHeadroomBytes = 512 * 1024 * 1024
```

Post-mlock, we query actual available bytes (free + speculative + inactive on macOS). If ≥ 512 MB remains, we keep the lock. If < 512 MB, we `munlock()` immediately.

This threshold was chosen empirically: it is large enough to prevent thrashing on 8 GB systems with working-set pressure from inference, and small enough not to block models that genuinely fit. A percentage would vary wildly across machines (512 MB is 6.4% of 8 GB but only 1.6% of 32 GB) and would break the same way on every small-memory machine. An absolute floor is the right invariant.

### Partial offload and mmap

When a model exceeds GPU layer capacity, Ollama offloads some layers to the CPU. The old code disabled `mmap` on Metal for any partial offload, forcing a full copy of weights into CPU ggml buffers in addition to the GPU copy. On an 8 GB machine with a 4–5 GB model, this pushes the working set into swap.

The fix removes the Metal-specific disable. `llama.cpp` already supports partial offload with `mmap` on Metal via the same `buffer_from_host_ptr` mechanism. Load times stay fast and swap pressure stays low.

### Prefix KV caching

The prefix cache saves K/V tensor rows for `num_keep` positions after the first forward pass. On subsequent loads, it restores those rows directly, skipping prompt recompute.

A headroom gate prevents restore when available RAM is below 300 MB. This is a safety valve: hydrating a large prefix cache under memory pressure would do more harm than good. The gate uses the same absolute-floor logic as the `mlock` policy for consistency.

---

## Design Decisions

| Decision | Rationale |
|---|---|
| **Absolute headroom floor, not percentage** | Percentage thresholds fail on small-memory machines (8 GB) and are unnecessarily permissive on large ones (32 GB). A fixed floor is the only policy that behaves consistently across the Apple Silicon line. |
| **No model-size gates** | `gemma4:e2b` is not special. The same memory policy should apply to every model. If a model mlocks safely, it gets pinning. If not, it gets zero-copy fallback. |
| **No runtime knob for threshold** | The 512 MB value was chosen from observed system behavior on 8 GB Macs. Making it configurable would invite users to break the invariant without understanding the failure mode. |
| `OLLAMA_USE_HOST_PTR=0` still works | Explicit opt-out is preserved for debugging or for environments where zero-copy is known to misbehave. |
| **Validation gate after full allocation** | `ValidateWeights()` fires after `allocModel()` (KV cache + graph memory) but before the first inference request. This is the peak memory-pressure point; a validation failure here means inference would have corrupted silently. |
| **Prefix cache stored per key hash** | The cache key is `SHA256(modelPath, archParams, prefixTokens)`. `IsRecurrent` is part of `archParams` so hybrid and pure-attention models never share entries. |

---

## Relationship to Upstream Ollama

OllamaSi is a fork of [ollama/ollama](https://github.com/ollama/ollama) with a narrow scope: Apple Silicon memory-policy correctness and runtime performance.

- Upstream Ollama is an excellent project. This fork does not claim to replace it.
- The changes here are intended as reference implementations for upstream consideration.
- No new model architectures or inference kernels are introduced.
- The CLI, REST API, model library, and Python/JS bindings remain unchanged.

Where upstream behavior is correct, OllamaSi does not touch it.

---

## Build and Install

Requires macOS with Xcode command-line tools.

```bash
git clone https://github.com/OllamaSi/ollama
cd ollama
go generate ./...
go build .
```

To run the server:

```bash
./ollama serve
```

To run a model:

```bash
ollama run qwen3.5:4b
```

All standard Ollama commands and environment variables work unchanged.

### Environment variables

| Variable | Effect |
|---|---|
| `OLLAMA_USE_HOST_PTR=0` | Force standard copy-path weight loading (disables zero-copy). |
| `OLLAMA_USE_HOST_PTR=1` | Force zero-copy even if the backend would normally disable it. |

---

## Support

If this fork saves you time, compute, frustration or stress, please consider supporting my future endeavours!

- [Buy me a coffee](https://buymeacoffee.com/corefidelity)
- [GitHub Sponsors](https://github.com/sponsors/Core-Fidelity)

Thanks for reading this far down!

---

## License

OllamaSi is released under the same license as upstream Ollama: [MIT](LICENSE).

Upstream Ollama is copyright © Ollama, Inc. This fork retains all upstream attribution and license headers.

