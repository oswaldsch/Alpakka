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

alpakka does not fork ollama. It reads models off disk, supervises one
`llama-server` process, and speaks ollama's wire API back out. Its store is a
directory of GGUF files.

`llama-swap` was evaluated and rejected: it supervises llama-server well but is
OpenAI-only, with no `/api/tags`, `/api/chat` or ollama NDJSON. It would have saved the supervisor and left the hard part —
wire compatibility — unwritten.

## Design

llama-server runs with `--jinja` and applies the model's own chat template.
alpakka never renders a template; it reads the GGUF's only to display it in
`/api/show`. This matters: `chat_template_kwargs` exists
only on the jinja path, so that is where `reasoning_effort` lives. It also
reduces the hard problem to one translator between ollama's chat shape and
OpenAI's, and makes `/v1/*` a near-passthrough.

- `internal/gguf` — GGUF metadata and tensor-shape reader.
- `internal/store` — model roots: directories of GGUF files.
- `internal/hub` — pulling GGUFs from HuggingFace.
- `internal/config` — TOML profiles, and the request/process split.
- `internal/supervisor` — the llama-server child: launch, fit check, eviction.
- `internal/wol` — Wake-on-LAN magic packets for sleeping RPC nodes.
- `internal/translate` — ollama ⇄ OpenAI, the only place wire shapes live.
- `internal/api` — HTTP handlers, model routing, load queuing.

## Install

```bash
./setup.sh
```

Builds alpakka, installs it into `~/.local/bin`, finds the llama.cpp build on
this machine, creates `~/models` if needed, writes
`~/.config/alpakka/config.toml` pointing at both, installs a systemd user unit and starts it — then waits for
the API to answer before claiming success. Re-run it to pick up a new build.

It needs a `llama-server` from llama.cpp, recent enough for `--fit` and
`--spec-type`, and it checks for those flags before installing anything:
distro packages tend to be a year behind.

`--system` installs to `/usr/local/bin`, `/etc/alpakka` and
`/etc/systemd/system` instead. `--dry-run` prints everything it would do,
including the unit. Detection can be overridden with `--lib-dir`, `--backend`,
`--models` and `--listen`; `--uninstall` removes it all again. `--help` lists
the rest.

The checks it makes are the ones whose absence is expensive: that the service
user can actually read the model root, and that it can open `/dev/kfd` and
`/dev/dri`. Failing either, llama-server still starts — on CPU, at a few tokens
a second, silently.

## Running

By hand, without the service:

```bash
go build -o alpakka ./cmd/alpakka
./alpakka                      # reads ~/.config/alpakka/config.toml
./alpakka -listen 127.0.0.1:11434   # take over ollama's port
```

Point any ollama client at it:

```bash
OLLAMA_HOST=127.0.0.1:11435 ollama list
OLLAMA_HOST=127.0.0.1:11435 ollama run qwen3.8-27b:q3-k-xl "..."
```

## Models

Models live at `<root>/<name>/<tag>.gguf`, with `<tag>.mmproj.gguf` beside them
for vision and llama.cpp's `<tag>-00001-of-0000N.gguf` for a split model. The
name is the directory, so `~/models/qwen3.8-27b/iq3-xxs.gguf` is
`qwen3.8-27b:iq3-xxs`, and a bare `qwen3.8-27b` resolves when it is the only
tag. Everything `/api/show` reports comes out of the GGUF header; system
prompts and parameters come from `config.toml` rather than from the store.

`alpakka pull` fetches a GGUF from HuggingFace into the first root:

```bash
alpakka pull https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B-UD-IQ3_XXS.gguf
alpakka pull hf.co/unsloth/Qwen3.8-27B-GGUF/UD-IQ3_XXS
alpakka pull unsloth/Qwen3.8-27B-GGUF:UD-IQ3_XXS      # -> qwen3.8-27b:iq3-xxs
```

It resumes an interrupted download, fetches every part of a split model, offers
the repo's projector when there is one, and refuses to replace an existing tag
without `-f`. `HF_TOKEN` is used for gated repos.

`alpakka list` (or `ls`) and `alpakka ps` print what the running server can
serve and what it has loaded. They ask the server rather than the store, since
`-models` may point it at other roots than the config. `-host` picks another
server. `ps` also shows the layer offload and whether a response is generating,
and `ps -v` adds the KV cache, any weights on the CPU, uptime, and the runtime
settings llama-server was started with.

`/api/ps` carries that under an `alpakka` key beside ollama's own fields, which
ollama clients ignore: `runtime`, `fit`, `started_at`, `uptime_ms` and `busy`.

`alpakka rm` is the other half of `pull`, and works on the store directly:

```bash
alpakka rm qwen3.8-27b:iq3-xxs
```

It deletes every part of a split model and the tag's projector. A projector
named for the repo rather than the tag serves every tag beside it, so it goes
only with the last of them, and the directory goes once it is empty. It refuses
a model the server has loaded until `alpakka stop` has unloaded it, or `-f` is
passed. If another root holds the same name, `rm` says which copy is served now.

## Commands

`alpakka help` lists them. Everything except `pull`, `rm` and `version` talks
to the running server, at the configured listen address or `-host`.

```bash
alpakka show qwen3.8-27b:q3-k-xl           # architecture, quant, context, capabilities
alpakka show -template qwen3.8-27b         # the GGUF's chat template; -modelfile, -info
alpakka run qwen3.8-27b "why is the sky blue"
alpakka run -o reasoning_effort=low -verbose qwen3.8-27b    # interactive chat
git diff | alpakka run qwen3.8-27b "review this"            # the diff follows the prompt
alpakka stop                               # unload now rather than at keep_alive
alpakka stop --force                       # without waiting for a response in progress
alpakka logs -n 200                        # llama-server's stderr for the last load
alpakka version                            # alpakka, llama-server, and which flags it has
```

`run` takes any `options` key as `-o key=value`, values read as JSON where they
parse, so `-o num_ctx=65536` is a number and `-o reasoning_effort=low` needs no
quotes. A process-level option reloads, as it would from any client. The
model's reasoning goes to stderr and the answer to stdout, so a piped answer is
only the answer; `-hide-thinking` drops the reasoning. Text piped on stdin is
answered, after the prompt argument when there is one; with neither, it starts a chat, where
`/clear` forgets the conversation and Ctrl-C stops an answer without leaving.

`stop` unloads through `POST /alpakka/unload`, not ollama's `keep_alive: 0`
generate request, which alpakka would answer by loading the model first. Like a
reload, it waits for a response still generating to finish, and says so when
there is one: `stop` can take as long as that answer does. `--force` unloads at
once and cuts the response off. Its ollama client gets a stream with no final
`done` chunk, and `alpakka run` exits reporting the answer cut off, rather than
either passing a truncated answer as complete. `stop <model>` refuses when a
different model is loaded.

`logs` reads `GET /alpakka/logs?n=`, the last 400 lines of llama-server's
stderr at most. It is kept after the process exits, so it still explains a load
that failed or a model that was evicted. The header goes to stderr and the lines
to stdout, for grep.

`version` runs the configured `llama-server --version` and checks its `--help`
for the flags `setup.sh` checks: it exits non-zero when one alpakka passes on
every load is missing, and reports which optional settings the build cannot do.

## Configuration

`~/.config/alpakka/config.toml`. Config keys use the same names as the request
`options` object, so anything settable per request is settable as a default.

```toml
[server]
listen = "127.0.0.1:11435"
# Browser origins allowed to call the API. Omitted, alpakka mirrors ollama's
# default: any port on localhost, plus the app://, file://, tauri:// and
# extension schemes desktop clients present. Setting this replaces that list.
# origins = ["https://chat.example"]

[store]
# Searched in order; the first root holding a name serves it, and a shadowed
# copy is logged. Omitted, this is ["~/models"]. `alpakka pull` writes into
# the first root.
roots = ["~/models"]

[llama]
lib_dir = "/opt/llama.cpp/bin"   # the directory holding llama-server, required
backend = "."                    # ggml backend subdirectory, e.g. "vulkan"

[defaults]
num_ctx = 32768
cache_type_k = "q8_0"
cache_type_v = "q8_0"
keep_alive = "5m"

[models."qwen3.8-27b:q3-k-xl"]
spec_type = "draft-mtp"
spec_draft_n_max = 2
reasoning_effort = "low"
projector = false
```

### Options

These ride in ollama's existing `options` object. Ollama ignores keys it does
not recognise, so a request carrying them stays valid against both servers.

**Per request, no reload:** `reasoning_effort` (`low` / `medium` / `xhigh`),
`temperature`, `top_k`, `top_p`, `min_p`, `typical_p`, `repeat_penalty`,
`repeat_last_n`, `presence_penalty`, `frequency_penalty`, `mirostat`,
`mirostat_tau`, `mirostat_eta`, `seed`, `num_predict`, `num_keep`, `stop`.

**Process level, triggers a clean reload:** `num_ctx`, `cache_type_k`,
`cache_type_v`, `cache_type_k_draft`, `cache_type_v_draft`, `num_batch`,
`num_ubatch`, `load_mode`, `spec_type`, `spec_draft_n_max`, `spec_draft_n_min`, `num_gpu`,
`allow_partial_offload`,
`gpu_vram_cap_mib`,
`flash_attn`, `backend`, `projector`, `embeddings`, `pooling`,
`kv_stream_arena_mib`, `num_cpu_moe`, `override_tensor`, `moe_expert_cache`,
`moe_expert_cache_inserts`, `rpc_servers`, `device`, `tensor_split`,
`split_mode`, `main_gpu`, `no_kv_offload`.

A request whose process-level options differ from the running process reloads
it. That is logged and never silently deferred to the next cold start — but the
reload waits for any response still streaming from the old process to finish
first, because killing it there would truncate that answer with no error at
either end. The same reference stops the keep-alive evictor: a generation that
runs longer than `keep_alive` cannot have its own server unloaded underneath it,
and `keep_alive: 0` unloads when the request completes rather than during it.

`reasoning_effort` accepts only `low`, `medium` and `xhigh`. The Qwen3.8
template rejects anything else and silently promotes `high` to `xhigh`, so
alpakka rejects `high` outright rather than let it mean the opposite of what was
asked. Ollama's own `think: "high"` is mapped to `xhigh` explicitly.

`kv_stream_arena_mib` is llama-server's `--kv-stream-arena-mib`, out of tree at
the time of writing: the KV cache lives in pinned host RAM and pages through a
VRAM arena of that many MiB, so a context the card could not hold still runs.
Zero or unset is the ordinary behaviour. Upstream streams one sequence only, so
alpakka rejects it outright alongside a `parallel` other than 1 rather than
letting the second request find out. An MTP draft cache is not streamed and
shares no pool with the target context, so it still needs its own VRAM.
`setup.sh` reports whether the build has the flag but does not require it.

`cache_type_k_draft` / `cache_type_v_draft` set the MTP draft context's KV
type, which llama.cpp otherwise leaves at f16: 512 MiB at 128K on
Qwen3.8-27B. `num_batch` and `num_ubatch` are `-b` / `-ub`. A smaller ubatch
shrinks the compute buffers (27B: 256 frees ~450 MiB at no prefill cost), a
larger one speeds up MoE prefill (35B-A3B: 2048 is 2.4x with `load_mode =
"none"`). `load_mode` is `--load-mode`: `auto`, `none`, `mmap`, `mlock`,
`mmap+mlock` or `dio`.

`num_cpu_moe` is `--n-cpu-moe`: the MoE expert weights of the first N layers
stay on the CPU, which trades their bandwidth for the VRAM to hold everything
else. `override_tensor` is `--override-tensor`, a list of `pattern=buffer`
entries for placing named tensors by hand:

```toml
num_cpu_moe = 12
override_tensor = ['blk\.[0-9]*[13579]\.ffn_.*_exps=CPU']
```

Entries are passed as one comma-separated flag. llama.cpp still accepts the flag
repeated but warns that the form is deprecated, and the warning would show up in
every load diagnosis.

`rpc_servers` is `--rpc`: a list of `"host:port"` llama.cpp RPC backend
endpoints, each one another device the automatic layer split can land on.
Needs no `--split-mode`, since layer split across whatever devices are present
is already llama.cpp's default. `rpc-server` has no auth or encryption, so
every entry must be LAN-only.

```toml
rpc_servers = ["192.168.178.62:50052"]
```

`device`, `tensor_split`, `split_mode`, `main_gpu` and `no_kv_offload` are
`--device`, `--tensor-split`, `--split-mode`, `--main-gpu` and
`--no-kv-offload`, for choosing which local GPUs a model spans. `device` is a
list or one comma-separated string of names from `llama-server --list-devices`;
the order matters, and it also numbers the entries `tensor_split` and
`main_gpu` refer to. `split_mode` is `none`, `layer`, `row` or `tensor`.
llama.cpp cannot place the KV cache apart from the layers: each layer's cache
lives on the GPU holding that layer, so `tensor_split` is how KV moves between
cards, and `no_kv_offload` is the only way to put it in host RAM (at a large
decode cost). Anything left unset passes nothing.

```toml
device = ["Vulkan1", "Vulkan0"]
tensor_split = "12,6"
```

`gpu_vram_cap_mib` is a ceiling on the sum over all GPUs the process uses. The
fit check counts layers over all devices, so a split load must still report
`N/N`.

Neither of the MoE options is the spill the fit check refuses. `offloaded N/M layers to
GPU` is computed from `-ngl` against the layer count and does not see per-tensor
placement, so a load steered by these still has to report `N/N`. What did land
on the CPU is reported in the ready line.

While any model is offloaded onto an RPC node, alpakka holds a `systemd-inhibit`
sleep and idle lock for as long as the process runs: that node stays reachable
only as long as this machine's own network does, and a suspend here would
silently stall every request it is carrying. The lock is released the moment
the process is torn down, on reload or eviction alike.

`wol`, outside `[defaults]` and `[models.*]`, maps an RPC endpoint's
`"host:port"` — the same string used in `rpc_servers` — to the MAC address of
the machine behind it:

```toml
[wol]
"192.168.178.62:50052" = "aa:bb:cc:dd:ee:ff"
```

A load that offloads onto a configured endpoint sends its MAC a Wake-on-LAN
magic packet first, broadcast on the LAN alongside `rpc_servers` itself. An
endpoint with no MAC configured here is assumed already running, exactly as
before this existed. alpakka does not wait for the node to come up: a load that
hits one still booting fails with llama-server's own connection error, the same
as it always has.

`moe_expert_cache` and `moe_expert_cache_inserts` are `--moe-expert-cache` and
`--moe-expert-cache-inserts` from llama.cpp PR #27861, an unmerged draft: no
released build has them, so this needs a llama-server built from that branch. It
keeps a VRAM LRU of the experts `num_cpu_moe` or `override_tensor` put in host
memory, and only on the decode path — prefill, batched decode and speculative
validation take the ordinary route. Without experts on the host there is nothing
to cache. Unset it passes nothing, because a build without the flags exits on
the unknown argument rather than ignoring it; `setup.sh` reports whether they
are there.

## Measuring a setting

`POST /alpakka/bench` runs a real generation and reports llama.cpp's own
timings, so a setting can be tried without editing the config and restarting.
It sits on alpakka's own prefix rather than `/api/*` or `/v1/*`, so no ollama or
OpenAI client can reach it.

```bash
curl -s localhost:11435/alpakka/bench -d '{
  "model": "qwen3.8-27b:q3-k-xl",
  "options": {"spec_type": "draft-mtp", "spec_draft_n_max": 2, "num_predict": 300},
  "prompt_tokens": 2048,
  "runs": 3
}'
```

`options` is the object `/api/chat` takes, so everything above is benchmarkable
and a setting added there needs nothing here. Settings that differ from the
running process reload it, which is the point; the reload's cost comes back as
`load_ms` and `reloaded`, because a configuration that reloads on every request
pays that every time.

`prompt_tokens` synthesises a prompt of about that length — words drawn at
random, so predictable filler cannot flatter a draft model, and a fresh one per
run with `cache_prompt` off, so the second run's prefill is measured rather than
served out of the first run's cache. `prompt` sends literal text instead. The
budget is `options.num_predict`, default 128, and runs decode with `ignore_eos`
so each produces the same count; `"ignore_eos": false` lets the model stop.

Each run reports `prompt_*` for prefill and `predict_*` for decode, in tokens,
ms and tokens/sec, taken from llama.cpp's counters rather than a wall-clock
delta around the stream. `runs` above one adds an `aggregate` pooling them,
which is worth having: the first run on a fresh process is always the slow one.
`runtime` and `fit` report what actually ran and what landed where, so a result
is self-describing. `/api/ps` reports those two for whatever is loaded, under
its `alpakka` key, without touching it.

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

The explicit exception is `allow_partial_offload = true`, which requires a
finite `num_gpu` and permits only that profile to keep the remaining whole
layers on the CPU. This is for a named oversized-model profile with a measured
VRAM budget; defaults and every profile without the opt-in stay strict.
`gpu_vram_cap_mib` measures the complete llama-server allocation from Linux DRM
fdinfo after load and refuses to serve it if the cap is exceeded or unverifiable.
Unlike `num_gpu`, this includes KV, graphs, vision, and allocator overhead.

The check fails closed. If llama-server does not print the offload line at all —
a reworded log message, a build that ignores `-lv` — the load is refused rather
than passed, because a check that silently stops running is worse than no check:
the symptom is exactly the slowdown it exists to catch.

Requests that arrive while a model is loading block until it is ready rather
than erroring. Eviction mirrors ollama: a five-minute default keep-alive,
honouring the request's `keep_alive`.

## Scope

Implemented: `/api/tags`, `/api/show`, `/api/ps`, `/api/version`, `/api/chat`,
`/api/generate`, `/api/embed`, `/api/embeddings`, and the `/v1` OpenAI surface
(`chat/completions`, `completions`, `embeddings`, `models`). Plus alpakka's own
`/alpakka/bench`, `/alpakka/unload` and `/alpakka/logs`,
on neither wire protocol.

`/v1/messages` and `/v1/messages/count_tokens` proxy llama-server's own
Anthropic endpoint, which is enough for Claude Code. llama-server forwards only
`temperature`, `top_p`, `top_k`, `stream` and `chat_template_kwargs` from it, so
profile `min_p` and `seed` have no effect there.

Embeddings get their own llama-server. llama.cpp refuses `/v1/embeddings` unless
the process was started with `--embeddings`, and refuses generation when it was,
so there is no process that does both and switching between them reloads. Like
ollama, any model can be embedded, not only a dedicated embedding model: the
endpoint decides, not the model's capabilities.

Two things follow, both handled. A causal model declares no pooling type, and
llama.cpp's OpenAI endpoint rejects the resulting `none` rather than returning
per-token vectors, so alpakka passes `--pooling last` — the reduction decoder
models are trained for — unless the GGUF names its own or `pooling` is set. And
`num_ctx` is capped at the context the model was trained for, since the chat
default would fail every load under `--fit off`.

Out of scope: `/api/pull`, `/api/create`, `/api/push`, `/api/copy`,
`/api/delete` return 501 pointing at `alpakka pull`. Adding or removing a model
is a command-line act (`alpakka pull`, `alpakka rm`), not an HTTP one. Also no auth and no concurrent models.

## Known divergences from ollama

- **Capabilities follow `/api/show`, not `/api/tags`.** Ollama's two endpoints
  disagree: `/api/tags` answers from a cache built at pull time, while
  `/api/show` computes live. For `gemma4:e4b`, show reports audio and vision and
  tags reports neither. alpakka computes both from the GGUF, like `/api/show`,
  because that is the endpoint clients query before deciding to send tools.
- **`context_length` is reported where ollama reports zero.** Ollama's metadata
  reader gives up on the gemma4 omni models, whose headers carry `vision.*` and
  `audio.*` sub-configs. The keys are present and correct, so alpakka reports
  them.
- **Every prompt is rendered by the GGUF's jinja template.** Ollama renders some
  families (gemma4, qwen3.5, qwen3.6) with built-in Go renderers, which may
  format prompts differently. Reimplementing ollama's renderer registry
  would be the permanent rebase tax this project exists to avoid.
- **`/api/generate` does not return `context`.** The token-id conversation
  handle has no equivalent on llama.cpp's OpenAI surface. It is deprecated in
  ollama and omitted here.

## Testing

```bash
go test ./...
```

Wire compatibility is checked against real captures, not recollection:
`internal/translate/testdata` holds both ollama's NDJSON and the target
llama-server's SSE. The supervisor tests drive a real `llama-server` and assert
the process is genuinely gone after a reload. They use the llama.cpp build and
the smallest model from the installed `config.toml` (or `ALPAKKA_LLAMA_LIB_DIR`
and `ALPAKKA_LLAMA_BACKEND`), and skip without one.

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
