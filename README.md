# alpakka

A local model server that speaks ollama's API so existing tools keep working,
but drives `llama-server` with performance settings ollama cannot express.

Measured on a Radeon RX 9060 XT 16 GB (gfx1200), ROCm 7.2.4, Ryzen 7 5700X:

| server | code | freeform |
|---|---|---|
| ollama as shipped, 32k ctx | 15.3 (spilling 718 MiB to CPU) | — |
| raw llama-server, MTP n=2 | 29.44 | 26.57 |
| **alpakka, same settings** | **29.28** | **26.43** |

Decode tokens/sec, greedy, 300 tokens, prompts from `gfx1200-lab`. alpakka
costs about half a percent against driving llama-server by hand, and roughly
doubles ollama.

Separately, `reasoning_effort` is a per-request setting: no reload, no restart.

## Why

Two large wins are unreachable through ollama. There is no flag, no environment
variable and no Modelfile parameter for either:

- **`--spec-type draft-mtp`.** The working model's GGUF ships a multi-token
  prediction head that llama.cpp will use as a built-in draft model. Memory
  bandwidth is ~320 GB/s and that is the ceiling on decode; speculative decoding
  is the only lever that beats a bandwidth wall, because it verifies several
  tokens per weight-read pass.
- **`chat_template_kwargs`.** Dropping the template's reasoning effort from
  `xhigh` to `low` took one coding request from 62 s to 18 s.

alpakka does not fork ollama. It reads ollama's model store read-only, supervises
one `llama-server` process, and speaks ollama's wire API back out.

`llama-swap` was evaluated and rejected: it supervises llama-server well but is
OpenAI-only, with no `/api/tags`, `/api/chat` or ollama NDJSON, and it does not
read ollama's store. It would have saved the supervisor and left the hard part —
wire compatibility — unwritten.

## Design

llama-server runs with `--jinja` and applies the model's own chat template.
alpakka never renders a template; it reads ollama's template layer only to
display it in `/api/show`. This is load-bearing: `chat_template_kwargs` exists
only on the jinja path, so that is where `reasoning_effort` lives. It also
reduces the hard problem to one translator between ollama's chat shape and
OpenAI's, and makes `/v1/*` a near-passthrough.

- `internal/gguf` — GGUF metadata and tensor-shape reader.
- `internal/store` — read-only view of `/var/lib/ollama/.ollama/models`.
- `internal/config` — TOML profiles, and the request/process split.
- `internal/supervisor` — the llama-server child: launch, fit check, eviction.
- `internal/translate` — ollama ⇄ OpenAI, the only place wire shapes live.
- `internal/api` — HTTP handlers, model routing, load queuing.

## Running

```bash
go build -o alpakka ./cmd/alpakka
./alpakka                      # reads ~/.config/alpakka/config.toml
./alpakka -listen 127.0.0.1:11434   # take over ollama's port
```

Point any ollama client at it:

```bash
OLLAMA_HOST=127.0.0.1:11435 ollama list
OLLAMA_HOST=127.0.0.1:11435 ollama run qwen3.8-27b-q3-32k "..."
```

## Configuration

`~/.config/alpakka/config.toml`. Config keys use the same names as the request
`options` object, so anything settable per request is settable as a default.

```toml
[server]
listen = "127.0.0.1:11435"

[llama]
lib_dir = "/usr/local/lib/ollama"
backend = "rocm_v7_2"          # or "vulkan"

[defaults]
num_ctx = 32768
cache_type_k = "q8_0"
cache_type_v = "q8_0"
keep_alive = "5m"

[models."qwen3.8-27b-q3-32k"]
spec_type = "draft-mtp"
spec_draft_n_max = 2
reasoning_effort = "low"
projector = false
```

### Options

These ride in ollama's existing `options` object. Ollama ignores keys it does
not recognise, so a request carrying them stays valid against both servers.

**Per request, no reload:** `reasoning_effort` (`low` / `medium` / `xhigh`),
`temperature`, `top_k`, `top_p`, `min_p`, `repeat_penalty`, `seed`,
`num_predict`, `stop`.

**Process level, triggers a clean reload:** `num_ctx`, `cache_type_k`,
`cache_type_v`, `spec_type`, `spec_draft_n_max`, `spec_draft_n_min`, `num_gpu`,
`flash_attn`, `backend`, `projector`.

A request whose process-level options differ from the running process reloads
it. That is logged and never silently deferred to the next cold start.

`reasoning_effort` accepts only `low`, `medium` and `xhigh`. The Qwen3.8
template rejects anything else and silently promotes `high` to `xhigh`, so
alpakka rejects `high` outright rather than let it mean the opposite of what was
asked. Ollama's own `think: "high"` is mapped to `xhigh` explicitly.

## Fitting is enforced

One 16 GB card, one model resident. alpakka passes `--fit off` so llama.cpp
cannot quietly shrink the context or the offload to make a load succeed, and
after every load it checks that llama.cpp offloaded every layer. A model that
did not fully fit is killed and the request fails with the reason:

```
llama-server exited during load:
E ggml_backend_cuda_buffer_type_alloc_buffer: allocating 13285.50 MiB on device 0: cudaMalloc failed: out of memory
E llama_init_from_model: failed to initialize the context: failed to allocate buffer for kv cache
```

A partial CPU spill is never served. That is the ollama behaviour this exists to
escape: it looks like a five-fold performance regression rather than a
misconfiguration.

Requests that arrive while a model is loading block until it is ready rather
than erroring. Eviction mirrors ollama: a five-minute default keep-alive,
honouring the request's `keep_alive`.

## Scope

Implemented: `/api/tags`, `/api/show`, `/api/ps`, `/api/version`, `/api/chat`,
`/api/generate`, `/api/embed`, `/api/embeddings`, and the `/v1` OpenAI surface
(`chat/completions`, `completions`, `embeddings`, `models`).

Out of scope: `/api/pull`, `/api/create`, `/api/push`, `/api/copy`,
`/api/delete` return 501 pointing at `ollama pull`. Also no auth, no multi-GPU,
no concurrent models.

## Known divergences from ollama

- **Capabilities follow `/api/show`, not `/api/tags`.** Ollama's two endpoints
  disagree: `/api/tags` answers from a cache built at pull time, while
  `/api/show` computes live. For `gemma4:e4b`, show reports audio and vision and
  tags reports neither. alpakka matches `/api/show` for all 22 local models,
  because that is the endpoint clients query before deciding to send tools.
- **`context_length` is reported where ollama reports zero.** Ollama's metadata
  reader gives up on the gemma4 omni models, whose headers carry `vision.*` and
  `audio.*` sub-configs. The keys are present and correct, so alpakka reports
  them.
- **Models declaring a `renderer`/`parser` are rendered by llama.cpp.** Ollama
  renders these (gemma4, qwen3.5, qwen3.6 in this store) with a built-in Go
  renderer. alpakka hands them to the GGUF's own jinja template instead, which
  may format prompts differently. Reimplementing ollama's renderer registry
  would be the permanent rebase tax this project exists to avoid.
- **`/api/generate` does not return `context`.** The token-id conversation
  handle has no equivalent on llama.cpp's OpenAI surface. It is deprecated in
  ollama and omitted here.

## Testing

```bash
go test ./...
```

Wire compatibility is checked against real captures, not recollection:
`internal/store/testdata` holds `/api/tags` and `/api/show` from a live ollama,
and `internal/translate/testdata` holds both ollama's NDJSON and the target
llama-server's SSE. The supervisor tests drive a real `llama-server` and assert
the process is genuinely gone after a reload.

Verified against the official `ollama` CLI and Python client, not only curl.

## Two traps worth inheriting

1. **`cd` into the backend directory before exec'ing llama-server.** ggml looks
   for `libggml-hip.so` in the executable's directory and the working directory.
   Miss it and the server starts happily on CPU at ~3 tok/s with no error.
2. **Never let a shell wrapper own the server process.** A kill that hits the
   wrapper leaves the real server holding the port and the VRAM. alpakka execs
   the binary directly and signals the process group.

## Naming

`Alpakka` is an established Akka Streams connector library. Fine for a personal
project; worth knowing before this goes anywhere public.
