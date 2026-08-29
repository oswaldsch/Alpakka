# alpakka — design

A local model server that speaks ollama's wire API while driving `llama-server`
with performance settings ollama cannot express.

## Why

Measured on the target machine (Radeon RX 9060 XT 16 GB, gfx1200, ROCm 7.2.4):

| config | decode tok/s |
|---|---|
| ollama as shipped, 32k ctx | 15.3, spilling 718 MiB to CPU |
| llama-server, all layers on GPU | 17.6 |
| + `--spec-type draft-mtp --spec-draft-n-max 2` | 25–31 |

Separately, `reasoning_effort: low` instead of `xhigh` took one coding request
from 62 s to 18 s. Ollama exposes neither knob: no flag, no env var, no
Modelfile parameter.

Memory bandwidth is ~320 GB/s and that is the decode ceiling. Speculative
decoding is the only lever that beats it. Nothing else in this design is a
performance optimisation.

## Rejected alternatives

**Forking ollama.** The change is new parameters threaded through a scheduler
that churns constantly. A permanent rebase tax for a feature upstream may never
want.

**llama-swap.** It supervises llama-server processes with full flag control,
has `ttl` unloading and per-model YAML `cmd:` lines. But it is OpenAI-only: no
`/api/tags`, `/api/chat`, `/api/show`, `/api/ps`, no ollama NDJSON framing. It
does not read ollama's model store, so all 22 models become hand-written YAML.
It would save the supervisor and leave the hard part — ollama wire
compatibility — entirely unwritten.

## Architecture

alpakka proxies to a single `llama-server` child process over that server's
**OpenAI-compatible API**, not its `/completion` endpoint.

llama-server runs with `--jinja` and applies the GGUF's own embedded chat
template. alpakka never renders a chat template; it reads ollama's `.template`
blob only to display it in `/api/show`. This is load-bearing:

- `reasoning_effort` rides on `chat_template_kwargs`, which only exists on the
  jinja path. Verified against a live server via `/apply-template`: both `low`
  and `xhigh` re-render the system prompt correctly, **per request, with no
  process restart**.
- `/v1/*` becomes a near-passthrough reverse proxy.
- The one hard problem shrinks to a single translator between ollama's
  `/api/chat` shape and OpenAI's.

### Packages

- **`store`** — read-only over `/var/lib/ollama/.ollama/models`. Parses
  manifests, resolves layers to blobs by digest, exposes model path, params
  JSON, template and system text. Never writes.
- **`supervisor`** — owns exactly one `llama-server` child: launch, readiness,
  teardown, eviction.
- **`translate`** — ollama ⇄ OpenAI request and response shapes, streaming and
  non-streaming. The only place wire formats live.
- **`api`** — HTTP handlers, model routing, load queuing.

### Two traps, inherited not rediscovered

1. `cd` into the backend directory (`/usr/local/lib/ollama/rocm_v7_2`) before
   exec'ing. ggml looks for `libggml-hip.so` in the executable's dir and the
   cwd; miss it and the server runs on CPU at ~3 tok/s with no error.
2. `exec` the server, and kill by process group. Otherwise the kill hits a
   wrapper shell and the real server keeps the port and the VRAM.

Both belong to `supervisor` and get a test each.

## Options

Split by cost. This split is the config's spine.

**Per-request, no reload:** `reasoning_effort`, all sampling parameters
(`temperature`, `top_k`, `top_p`, `seed`, `num_predict`, `stop`).

**Process-level, requires reload:** model, `num_ctx`, `cache_type_k`,
`cache_type_v`, `spec_type`, `spec_draft_n_max`, `num_gpu`, projector on/off.

New keys ride in ollama's existing `options` object, which ollama itself
ignores when unknown — so requests stay compatible in both directions.

A request whose process-level options differ from the running process triggers
an explicit teardown and reload. Never a silent no-op deferred to the next cold
start. The reload is logged and visible in `/api/ps`.

## Fit is a hard error

`-ngl 99` always. After load, check `/props` and confirm every layer landed on
GPU. If not, kill the process and fail the request with a clear error. A
partial CPU spill is never served — that is the ollama behaviour this project
exists to escape.

At 64k context with q4_0 cache and MTP on, the working model sits at ~16.1 GB
of 16.3 GB. There is no headroom to absorb a mistake.

## Loading and eviction

One 16 GB card, one model resident. Model switching is a teardown and reload,
and reloads are slow — tens of seconds for 13 GB.

Requests arriving during a load block on a readiness channel with a generous
timeout rather than erroring. Ready means `/health` returns ok **and** the fit
check passes.

Eviction mirrors ollama: 5-minute default keep-alive, honouring the request's
`keep_alive` field.

## Endpoints

**v1 scope:** `/api/tags`, `/api/show`, `/api/ps`, `/api/generate`,
`/api/chat`, `/api/embeddings`, and the `/v1/*` OpenAI surface that opencode
drives today. Streaming NDJSON must match ollama's exact shape.

**Explicitly out of scope:** `/api/pull`, `/api/create`, `/api/push`,
`/api/copy`, `/api/delete` return 501 with a message directing the caller to
`ollama pull`. Failing cleanly beats half-working.

Also out: auth, multi-GPU, concurrent models.

## Config

TOML at `~/.config/alpakka/config.toml`. A `[defaults]` block plus per-model
overrides. Listen address is a config key.

```toml
[server]
listen = "127.0.0.1:11435"

[defaults]
ctx = 32768
cache_type_k = "q8_0"
cache_type_v = "q8_0"
reasoning_effort = "low"
keep_alive = "5m"

[models."qwen3.8-27b-q3-32k"]
spec_type = "draft-mtp"
spec_draft_n_max = 2
projector = false
```

`reasoning_effort` accepts `low`, `medium`, `xhigh` only; the Qwen3.8 template
rejects anything else and silently upgrades `high` to `xhigh`.

## Verification

- `translate` is unit-tested against NDJSON captured from the real ollama on
  `:11434`, so golden files are ground truth rather than recollection.
- End-to-end against a real ollama client, not curl. Wire bugs live in
  streaming framing and field names, which curl does not catch.
- `~/Local_LLMs/Benchmarks/gfx1200-lab/bench.sh` run through alpakka against
  raw `serve.sh`. Same prompts, greedy, fixed seed — MTP is lossless under
  greedy, so output should be byte-identical. Slower than that repo's README
  numbers is a bug in alpakka.

## Naming

`Alpakka` is an established Akka Streams connector library. Fine for a personal
project; worth knowing before it goes public.
