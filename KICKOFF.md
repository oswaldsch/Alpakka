# alpakka — kickoff brief

Paste this into a fresh Claude Code session started in `~/code/alpakka`.

---

I want to build **alpakka**: a local model server that speaks ollama's API so my
existing tools keep working, but drives llama.cpp with performance settings that
ollama simply cannot express.

Start by reading this whole brief, then push back on anything you think is wrong
before writing code. The architecture call in particular is not settled.

## Why this exists

I benchmarked my setup and found two large wins that ollama structurally cannot
give me. Numbers are measured on my machine, not estimates:

| config | decode tok/s |
|---|---|
| ollama as shipped, 32k ctx | 15.3 (and it spilled 718 MiB of weights to CPU) |
| llama-server, same model, all layers on GPU | 17.6 |
| + `--spec-type draft-mtp --spec-draft-n-max 2` | **25–31**, workload dependent |

Separately, dropping the chat template's reasoning effort from `xhigh` to `low`
cut a small coding request from **62 s to 18 s** (1676 → 523 completion tokens)
with no meaningful quality loss.

Neither is reachable through ollama. I checked its binary and its full env var
list: there is no flag, no env var and no Modelfile parameter for `--spec-type`
or for chat-template kwargs. That gap is the entire reason for this project.

## Hardware and paths

- Radeon RX 9060 XT 16 GB (Navi 44, **gfx1200**), ROCm 7.2.4, Ryzen 7 5700X, Arch.
- **~320 GB/s memory bandwidth.** This is the real ceiling on decode speed;
  speculative decoding is the only lever that beats it. Keep that in mind before
  optimising anything else.
- Fast llama.cpp: `/usr/local/lib/ollama/llama-server`, backends in
  `/usr/local/lib/ollama/{rocm_v7_2,vulkan}/`.
- **Do not use `/usr/bin/llama-server`** (Arch `llama-cpp` package). It is a
  Vulkan build with asserts enabled and prints as much on startup.
- Ollama's model store: `/var/lib/ollama/.ollama/models/{blobs,manifests}`.
  Standard OCI-ish manifests; layers carry mediaTypes `...image.model`,
  `.template`, `.projector`, `.params`, `.system`.

## Two traps that already cost me hours — inherit them, don't rediscover them

1. **You must `cd` into the backend directory before exec'ing llama-server.**
   ggml looks for `libggml-hip.so` in the executable's dir and the cwd. Miss it
   and the server starts happily on CPU at ~3 tok/s with no error — it looks
   like a 5x perf regression, not a misconfiguration.
2. **`exec` the server inside any subshell you background.** Otherwise `$!` is a
   wrapper shell, your kill hits the wrapper, and the real server keeps the port
   and the VRAM. This silently made one of my benchmarks measure the *previous*
   config and produce a completely wrong conclusion.

## The architecture question — decide this first

I do not know whether to fork ollama or build something new. My current thinking,
which I want you to challenge:

**Probably don't fork.** Ollama is a large, fast-moving Go codebase and the
change I need is threading new parameters through its scheduler into the
llama-server invocation. That is a small diff living in a place that churns
constantly — a permanent rebase tax for a feature upstream may never want.

**Probably do reuse its model store.** I have 20+ models already pulled. The
manifests are plain JSON and resolve to blobs by digest, including the chat
template and params. Reading that store read-only gets me model management,
Modelfile templates and my existing library for almost free, without
reimplementing `/api/pull` or a blob downloader on day one.

So: a new server that reads ollama's store, spawns and supervises `llama-server`
processes with per-model flag profiles, and speaks ollama's wire API back out.

**Before committing, check whether `llama-swap` already does most of this.** It
proxies to llama-server instances with full flag control. If it does, the honest
answer might be "configure llama-swap and write a thin ollama-API adapter", and
I would rather know that than build something redundant. Tell me what you find.

## Compatibility target

Full wire compatibility with ollama clients for the endpoints that actually get
used. Prioritise:

- `/api/tags`, `/api/show`, `/api/ps` — so tooling can enumerate models
- `/api/generate`, `/api/chat` — including **streaming NDJSON in ollama's exact
  shape**, which is where naive reimplementations break
- `/api/embeddings`
- `/v1/*` OpenAI-compatible surface (opencode drives this today)

`/api/pull`, `/api/create`, `/api/push` are explicitly out of scope for v1. If a
client calls them, fail cleanly with a clear message rather than half-working.

## The new options

The natural extension point is ollama's existing `options` object on
`/api/generate` and `/api/chat`. Unknown keys there are ignored by ollama itself,
so adding ours keeps us wire-compatible in both directions:

- `spec_type` (e.g. `draft-mtp`), `spec_draft_n_max`
- `reasoning_effort` (`low` / `medium` / `xhigh` for Qwen3.8; the template
  rejects anything else and silently upgrades `high` to `xhigh`)
- explicit `cache_type_k` / `cache_type_v`
- anything else needed to stop the auto-fit logic quietly moving layers to CPU

Also support a config file of per-model defaults, so I do not have to send these
on every request. Changing a flag that requires a different llama-server process
must trigger a clean reload, not silently apply on the next cold start.

## Constraints that will bite

- **One 16 GB card, one model resident at a time.** Model switching means tearing
  down and reloading, and reloads are slow. Keep-alive and eviction need to be
  deliberate, not an afterthought.
- At 64k context with q4_0 cache and MTP on, my working model sits at ~16.1 GB of
  16.3 GB. There is almost no headroom. **Failing to load must be a clean error,
  never a partial CPU-spill that silently destroys performance** — that is
  precisely the ollama behaviour I am trying to escape.
- Health/readiness matters: llama-server takes tens of seconds to load 13 GB.
  Requests arriving during load must queue, not error.

## How I want you to work

- Talk through the design with me before building. I will push back.
- I have a working benchmark harness at
  `~/Local_LLMs/Benchmarks/gfx1200-lab` (`./bench.sh`, `./report.sh`, results in
  `results/results.csv`). **Use it to prove alpakka does not regress against raw
  llama-server.** If alpakka is slower than the numbers in that repo's README,
  that is a bug in alpakka.
- Verify against a real ollama client, not just curl. Wire-compatibility bugs
  live in streaming framing and field names, which curl will not catch.
- I would rather have a small thing that is correct than a broad thing that is
  approximately right.

## Naming note

`Alpakka` is already a well-known Akka Streams connector library. Fine for a
personal project, worth knowing if this ever goes public.
