# bench autotuner — future feature, not yet implemented

Drive `/alpakka/bench` automatically to find a good process-level
configuration for a model on this hardware, instead of hand-editing
`config.toml` and re-running curl by hand.

## Why

`/alpakka/bench` already reports real llama.cpp-measured tokens/sec for any
combination of `num_ctx`, `cache_type_k/v`, `flash_attn`, `spec_type` +
`spec_draft_n_max/min`, `num_cpu_moe`, `override_tensor`, `moe_expert_cache` +
`moe_expert_cache_inserts`, and `kv_stream_arena_mib`. Finding a good point in
that space by hand means guessing a config, waiting out a reload, reading a
number, and repeating. That's the part worth automating.

Two things make this a real search problem rather than a sweep:

- **Evaluations are expensive and reload-dominated.** Any process-level
  option change costs a real 10–60+ second `llama-server` reload on this
  hardware, not milliseconds. Time spent reloading dwarfs time spent
  measuring, so the search has to economize on reloads specifically, not on
  measurements.
- **A candidate can be flatly infeasible, not just slow.** `checkFit`
  (`internal/supervisor/instance.go`) refuses to serve a partially-offloaded
  model. A bad config comes back as a load error through the existing
  diagnosis machinery, not a bench result — the search has to treat
  "infeasible" as information to route around, not noise to retry past.

## Rejected alternative: full Bayesian optimization

A first design pass (GPT-6 Astra, asked to plan this in isolation) proposed
three averaged Gaussian-process regressions (log prefill ms/token, log decode
ms/token, per workload), a Laplace-approximated probit GP classifier for
feasibility, and a two-step Monte-Carlo-simulated expected-utility-per-second
bandit policy over proposal pools of up to 512 candidates, with 16×8 rollout
simulations per decision.

Technically coherent, and it found real bugs by reading the actual source
(see "Bugs found along the way" below) — but the optimization engine itself
is disproportionate to a budget of roughly 30 process starts on one GPU for
one person. Implementing it faithfully would be the single most complex
subsystem in the repo, would need an actual ML stack, and directly
contradicts this project's own design stance: small, explicit, fail loud
rather than model around uncertainty. Rejected on those grounds, not on
correctness.

## Design: greedy search, not a surrogate model

**Unit of work is a process visit, not a single measurement.** Load one
configuration, run every cheap (no-reload) variation and repeat worth having
while it's resident, then decide whether to reload into something else. This
is the one idea worth keeping from the rejected design.

**Reuse `config.Profile`'s own process-level/request-level split** as the
axis that decides what "cheap while resident" means — see the field
comments in `internal/config/config.go`. Only process-level fields ever
justify a reload; sampler/request-level fields (already excluded from
throughput-relevant tuning) never do.

**Canonicalize before comparing configs**, mirroring how `Supervisor.Ensure`
compares the *whole* resolved `Runtime`, not just active flags. Two configs
that differ only in an inactive field's representation (e.g.
`moe_expert_cache_inserts` set vs. unset while the cache itself is off) are
the same config and must not trigger a reload or count as two search points.

**Two workload families per context tier**, not one: a short prompt (fixed
length, same across tiers) and a long prompt sized to ~75% of the tier under
test. This separates "cost of allocating a bigger context" from "cost of
actually filling it" — collapsing them onto one throughput number hides
which one a given setting actually affects.

**Search strategy: greedy coordinate descent from a sane starting config**
(current `config.toml` defaults for the model), moving one dimension at a
time toward better measured throughput, with **bisection when a boundary
flips between feasible and infeasible** (arena-too-small vs. arena-OOM,
`num_cpu_moe` vs. VRAM headroom) rather than a fixed step size. No surrogate
model, no acquisition function — the reload cost is the whole budget
constraint, and a greedy walk spends it directly on configs that are likely
to matter instead of on modeling the space in the abstract.

**Failure taxonomy drives the next proposal**, reusing `checkFit`'s own
distinctions: an explicit OOM/allocation failure means "back off," an
arena-too-small diagnostic means "grow the arena," a partial-offload refusal
is a hard exclusion (never scored as a slow success), and an unrecognized
failure gets at most one retry before the search moves on rather than
looping on it.

**Stop when the last few tries haven't beaten the best result by some
threshold** (start with 3%, make it configurable) or the time/reload budget
runs out — whichever comes first. No confidence intervals, no posterior
convergence tests.

## Interface

Drives `/alpakka/bench` exactly as documented — this is a client of that
endpoint, not a change to it. No new alpakka internals beyond the tuner
itself (a separate tool/command, TBD whether it lives in this repo or
alongside it).

## Bugs found along the way (fix before or during implementation)

- `internal/api/bench.go`'s synthetic filler-prompt generator reseeds from
  the run index on every HTTP request, so repeated `runs:1` calls at the same
  offset silently reuse the same prompt. A tuner comparing repeated
  single-run calls needs this fixed first, or it needs to always ask for
  `runs >= 2` itself to avoid relying on prompt variety it isn't getting.
- `internal/config/options.go`'s `config.Apply` does not accept `parallel`
  as a per-request option key — it's settable only in `config.toml`. If the
  tuner ever needs to search over it, that gap has to close first; until
  then, treat `parallel` as fixed for the duration of a tuning run.

## Open questions

- Where this tool lives (new `cmd/` binary vs. a script) and whether it
  needs its own persistence for a long-running campaign, or can just print a
  frontier table and exit.
- Whether `override_tensor` recipes are ever synthesized automatically or
  always user-supplied — the bench endpoint has no view of tensor names or
  model architecture, so an automatic search cannot invent placement regexes
  from nothing.
