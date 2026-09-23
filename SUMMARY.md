# Alpakka operator summary

Read this before using, testing, or diagnosing Alpakka. `README.md` is the full
design and feature reference; this file is the compact mental model for agents
that need to operate it without first reading the codebase.

## What it is

Alpakka is an Ollama-compatible HTTP front end for a supervised `llama-server`.
It does **not** fork Ollama, render chat templates, or run an Ollama inference
process. Models are added only from the command line (`alpakka pull`,
`alpakka import --from-ollama`), never over HTTP. It:

1. reads models read-only from its roots: a `<name>/<tag>.gguf` directory
   and/or Ollama's on-disk store;
2. resolves a requested tag to its GGUF, template, projector, and metadata;
3. merges global, per-model, and per-request settings;
4. starts exactly one `llama-server` with the required process-level flags;
5. proxies OpenAI requests or translates Ollama requests and streams the result
   back in Ollama's NDJSON shape.

The point is to expose llama.cpp controls Ollama cannot: MTP speculative decode,
Jinja `chat_template_kwargs`, explicit KV quantization, strict no-spill loading,
optional RPC placement, and experimental KV/MoE controls.

## This machine

- GPU: Radeon RX 9060 XT 16 GB (16,304 MiB usable), ROCm, `gfx1200`.
- Alpakka: `127.0.0.1:11435`.
- Writable Ollama daemon/model management: `127.0.0.1:11434`.
- Config: `~/.config/alpakka/config.toml`.
- Service: `~/.config/systemd/user/alpakka.service`.
- Binary: `~/.local/bin/alpakka`.
- llama.cpp build: `~/code/llama.cpp/build/bin` in the current install.
- Ollama store: `/var/lib/ollama/.ollama/models`.

Use port 11435 for inference and inspection. Add plain GGUFs with
`alpakka pull`, or use port 11434 for `ollama pull` and `ollama create`.
Alpakka intentionally returns 501 for store mutations over HTTP.

```bash
OLLAMA_HOST=http://127.0.0.1:11435 ollama list
OLLAMA_HOST=http://127.0.0.1:11434 ollama pull gpt-oss:20b
```

Pulling through 11434 writes to disk and does not itself load or unload the
model being served by Alpakka.

## Configuration and precedence

The effective profile is layered in this order, last value wins:

```text
built-in defaults
  -> [defaults] in config.toml
  -> [models."exact-or-bare-name"] in config.toml
  -> request options
```

A tagged model can inherit a bare-name profile. Explicit request values win.

**Ollama Modelfile params are not part of this runtime merge.** Alpakka reads
the manifest params layer for `/api/show` and generated Modelfile output, but
sampling/runtime defaults used for inference come from `config.toml` and the
request. Therefore `PARAMETER temperature 0.3` may appear in `/api/show` without
making 0.3 the Alpakka default. Put it in TOML or send it per request:

```toml
[models."qwen3.8-27b-q3-64k:latest"]
temperature = 0.3
```

Options are request-level (no reload) or process-level (clean reload). The full
lists are under Options in `README.md`.

Changing any process-level value produces a different comparable runtime and
therefore reloads. A reload waits for active responses to finish; it never
truncates an in-flight stream.

The current Qwen profiles are 32K, 64K, and 128K tags with Q4_0 K/V cache,
`draft-mtp`, `spec_draft_n_max = 2`, low reasoning effort, and no projector.
Check the TOML rather than trusting this sentence after future tuning.

## Model lifecycle

There is one child `llama-server` and one resident model at most.

- Same model plus identical process settings: reuse the running server.
- Different model or process setting: wait for it to become idle, stop it,
  then start the requested runtime.
- Concurrent requests for one loading runtime share the same load and wait.
- A request holds a reference for its full lifetime, preventing reload/eviction.
- `keep_alive` starts again when a request **finishes**, not when it starts.
- A generation longer than `keep_alive` remains pinned and is not killed.
- `keep_alive <= 0` means unload after the request completes.
- Cold-load timeout is five minutes; generation has no server-side timeout.
- Client disconnect propagates cancellation to llama-server.

The child uses a free loopback port. It is executed directly in its own process
group, with its working directory set to the selected backend directory. Both
details are load-bearing: a shell wrapper can orphan the VRAM-owning process,
and a wrong cwd can make ROCm libraries disappear and silently run on CPU.

## Templates and reasoning

llama-server runs with `--jinja` and renders the GGUF's own template. Alpakka
does not render the Ollama template layer; `/api/show` displaying a template is
not proof that this is the template llama.cpp executes.

`reasoning_effort` is passed as a Jinja `chat_template_kwargs` value and accepts
only `low`, `medium`, or `xhigh`. Ollama `think: "high"` maps explicitly to
`xhigh`; boolean `think` controls `enable_thinking`. OpenAI clients may send
`chat_template_kwargs` directly, and their explicit value beats the profile.

Reasoning tokens are completion tokens. `num_predict`/`max_tokens` is a combined
ceiling for hidden thinking and the visible answer, not a separate thinking
budget. A model that consumes the whole limit inside `<think>` can finish with
`done_reason: length` and an empty visible response. That often looks like a
crash in clients such as OpenCode. Increasing the output limit only postpones
the failure; disabling thinking or using a backend-supported reasoning control
is the real mitigation.

Alpakka does not impose a generation-duration limit. A repeatable wall-clock
cutoff can be a token ceiling translated through decode speed, a client timeout,
or context exhaustion. Separate cold load and prompt prefill from decode before
diagnosing it: reprocessing 50K uncached tokens can take minutes before the
first generated token.

## VRAM, context, and deliberate refusal

Alpakka refuses to serve a partial CPU spill (see "Fitting is enforced" in
`README.md`). An OOM is preferable to Ollama silently moving weights to CPU and
losing most decode performance.

The displayed GGUF trained context (for example 262K) is not the configured
runtime context. The runtime is the resolved `num_ctx`. A 262K metadata value
does not mean that this 16 GB card can load it.

VRAM consists of more than the GGUF size:

```text
weights + target KV + MTP draft KV + compute/scratch + graphs + projector
        + ROCm allocation overhead + desktop compositor headroom
```

KV use grows roughly linearly with `num_ctx`. Q4 KV is about half Q8 KV. MTP's
draft cache remains additional VRAM. A projector reserves substantial VRAM even
if a request has no image, which is why `projector = false` is important here.
At 64K Q4 plus MTP this machine has only a few hundred MiB of headroom; normal
Plasma/Wayland allocation changes can decide whether a load fits.

Typical expected failure:

```text
llama-server exited during load:
... failed to allocate buffer for kv cache
... out of memory
```

That is a clean fit refusal, not evidence that Alpakka itself crashed. The
supervisor retains notable error lines and returns them to the client. It also
fails closed if llama.cpp stops printing the `offloaded N/N layers to GPU`
line, because otherwise a changed log format could disable spill detection.

On this AMD machine, inspect real VRAM ownership with:

```bash
amdgpu_top -p
```

Per-process rows may double-count shared Wayland buffers; total VRAM minus the
llama-server allocation is the reliable aggregate for desktop/headroom usage.

## Prompt caching

llama-server can reuse a resident slot's matching prompt prefix. This speeds
repeated chat histories while the same compatible runtime remains loaded. A
cold load, model/context change, process reload, changed prefix, or eviction can
force full prefill again. Cache reuse does not reduce newly generated token
counts and cannot bypass output limits.

## API surface and compatibility

Endpoints, embedding behavior and the known divergences from Ollama are under
Scope and Known divergences in `README.md`. Store mutations (`/api/pull`,
`/api/create`, `/api/push`, `/api/copy`, `/api/delete`) return 501: use
`alpakka pull`, or Ollama on 11434.

Ollama chat/generate requests are translated to llama-server's streaming
OpenAI chat endpoint. llama-server is always asked to stream; Alpakka either
forwards translated NDJSON chunks or collects them for `stream: false`.
`/api/generate` is treated as a single-turn chat so Jinja reasoning controls
remain active. It does not return Ollama's deprecated token-ID `context` field.

The `/v1` surface is mostly proxied. Alpakka resolves/loads the model, injects
missing configured sampling/template defaults, preserves explicit client
values, rewrites the alias, and forwards streaming SSE. Process-level Alpakka
options can be carried in a top-level `options` object.

## Diagnosis without generating

These calls inspect state and do not load a model:

```bash
curl -s http://127.0.0.1:11435/api/tags | jq
curl -s http://127.0.0.1:11435/api/ps | jq
curl -s http://127.0.0.1:11435/alpakka/status | jq
curl -s http://127.0.0.1:11435/api/show \
  -H 'Content-Type: application/json' \
  -d '{"model":"MODEL:TAG"}' | jq
systemctl --user --no-pager --full status alpakka
journalctl --user -u alpakka --since '15 minutes ago' --no-pager
```

Interpret common symptoms:

- Several minutes before first token: cold weight load and/or uncached prefill.
- Empty answer after long thinking: completion cap reached inside reasoning.
- HTTP 400 saying request exceeds context: prompt plus requested output does not
  fit `num_ctx`; increase context only if VRAM permits or shorten the request.
- Load-time OOM: requested runtime does not fit; lower context/KV precision,
  disable projector/speculation, or deliberately place supported tensors away.
- About 3 tok/s: likely wrong backend/cwd or no GPU access, not normal variance.
- Request waits while another streams: requested runtime differs, so reload is
  correctly waiting for the active request rather than killing it.
- Model disappears after idle time: normal `keep_alive` eviction.
- Browser-only failure with no useful server trace: inspect CORS origins.
- Embedding request causes reload: expected process-mode switch.

Do not diagnose a client-side “crash” from wall time alone. Check its output
limit and final reason, Alpakka's journal, `/alpakka/status`, and whether the
llama-server PID actually exited.

## Safe operating rules

- Do not benchmark by reading or copying sibling implementations first.
- Do not send a generation merely to inspect configuration or health.
- Do not point model-management commands at port 11435.
- Do not assume `/api/show` Modelfile params are effective Alpakka defaults.
- Do not change process-level options during a latency measurement; that adds a
  reload unless the running runtime already matches.
- Do not interpret `keep_alive = "5m"` as a five-minute request timeout.
- Do not enable a projector on a memory-tight text-only profile.
- Do not assume a model fits because weights alone fit; include all caches,
  scratch space, and desktop headroom.
- Do not bypass the all-layers fit check unless silent CPU spill is acceptable.
- Never kill only a wrapper process around llama-server; the supervisor uses a
  process group specifically to avoid orphaning the VRAM owner.

For implementation changes, benchmarks, experimental flags, install details,
and protocol divergences beyond normal operation, continue with `README.md` and
the relevant package. For ordinary use and incident triage, this file should be
enough.
