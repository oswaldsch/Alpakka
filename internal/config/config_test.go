package config

import (
	"os"
	"path/filepath"
	"testing"
)

func write(t *testing.T, body string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "config.toml")
	if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestLoadMissingFileUsesDefaults(t *testing.T) {
	cfg, err := Load(filepath.Join(t.TempDir(), "absent.toml"))
	if err != nil {
		t.Fatalf("a missing config should not be an error: %v", err)
	}
	if cfg.Server.Listen != "127.0.0.1:11435" {
		t.Errorf("listen = %q", cfg.Server.Listen)
	}
}

func TestLoadLayersOverDefaults(t *testing.T) {
	cfg, err := Load(write(t, `
[server]
listen = "127.0.0.1:11434"

[defaults]
reasoning_effort = "low"

[models."qwen3.8-27b-q3-32k"]
spec_type = "draft-mtp"
spec_draft_n_max = 2
projector = false
`))
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Server.Listen != "127.0.0.1:11434" {
		t.Errorf("listen = %q", cfg.Server.Listen)
	}
	// Untouched defaults must survive a partial file.
	if got := deref(cfg.Defaults.CacheTypeK, ""); got != "q8_0" {
		t.Errorf("cache_type_k = %q, want the default q8_0", got)
	}

	p := cfg.ForModel("qwen3.8-27b-q3-32k")
	if got := deref(p.SpecType, ""); got != "draft-mtp" {
		t.Errorf("spec_type = %q", got)
	}
	if got := deref(p.ReasoningEffort, ""); got != "low" {
		t.Errorf("reasoning_effort = %q, want the global default to carry through", got)
	}
}

// A request names a model as "name:tag"; the config file usually keys it by the
// bare name. Both must resolve to the same profile.
func TestForModelMatchesTaggedName(t *testing.T) {
	cfg, err := Load(write(t, `
[models."qwen3.8-27b-q3-32k"]
spec_draft_n_max = 2
`))
	if err != nil {
		t.Fatal(err)
	}
	if got := deref(cfg.ForModel("qwen3.8-27b-q3-32k:latest").SpecDraftNMax, 0); got != 2 {
		t.Errorf("spec_draft_n_max = %d, want 2", got)
	}
}

func TestRejectsBadReasoningEffort(t *testing.T) {
	// "high" is the dangerous one: the template silently promotes it to xhigh,
	// which is the slow path the caller was trying to avoid.
	if _, err := Load(write(t, "[defaults]\nreasoning_effort = \"none\"\n")); err != nil {
		t.Fatalf("reasoning_effort = none must be accepted: %v", err)
	}
	if _, err := Load(write(t, "[defaults]\nreasoning_effort = \"high\"\n")); err == nil {
		t.Fatal("expected reasoning_effort = high to be rejected")
	}
}

func TestRejectsBadKeepAlive(t *testing.T) {
	if _, err := Load(write(t, "[defaults]\nkeep_alive = \"soon\"\n")); err == nil {
		t.Fatal("expected an invalid keep_alive to be rejected")
	}
}

func TestPartialOffloadRequiresFiniteGPUCount(t *testing.T) {
	for _, body := range []string{
		"[defaults]\nallow_partial_offload = true\n",
		"[defaults]\nallow_partial_offload = true\nnum_gpu = 99\n",
	} {
		if _, err := Load(write(t, body)); err == nil {
			t.Errorf("expected unsafe partial-offload profile to be rejected: %q", body)
		}
	}
	if _, err := Load(write(t, "[models.oversized]\nallow_partial_offload = true\nnum_gpu = 18\n")); err != nil {
		t.Fatalf("explicit bounded partial offload was rejected: %v", err)
	}
}

func TestLoadCarriesWoLMACs(t *testing.T) {
	cfg, err := Load(write(t, `
[wol]
"192.168.178.62:50052" = "aa:bb:cc:dd:ee:ff"
`))
	if err != nil {
		t.Fatal(err)
	}
	if got := cfg.WoL["192.168.178.62:50052"]; got != "aa:bb:cc:dd:ee:ff" {
		t.Errorf("wol[...] = %q", got)
	}
}

func TestRejectsBadWoLMAC(t *testing.T) {
	if _, err := Load(write(t, "[wol]\n\"192.168.178.62:50052\" = \"not-a-mac\"\n")); err == nil {
		t.Fatal("expected an invalid MAC address to be rejected")
	}
}

func TestRejectsEUI64WoLMAC(t *testing.T) {
	// A valid net.ParseMAC input but not a 6-byte MAC, which the magic packet
	// format requires.
	if _, err := Load(write(t, "[wol]\n\"192.168.178.62:50052\" = \"aa:bb:cc:ff:fe:dd:ee:ff\"\n")); err == nil {
		t.Fatal("expected an 8-byte EUI-64 address to be rejected")
	}
}

func TestKeepAliveDefaultsToFiveMinutes(t *testing.T) {
	var p Profile
	if got := p.KeepAliveDuration().String(); got != "5m0s" {
		t.Errorf("keep-alive = %s, want ollama's 5m default", got)
	}
}

// The supervisor decides on reloads by comparing Runtime values, so equality
// has to reflect every process-level flag.
func TestRuntimeComparesProcessLevelFlags(t *testing.T) {
	base := Profile{}.Runtime("m", "/blob", "", false)
	if base != (Profile{}.Runtime("m", "/blob", "", false)) {
		t.Fatal("identical profiles produced unequal runtimes")
	}
	if base == (Profile{SpecDraftNMax: ptr(2)}.Runtime("m", "/blob", "", false)) {
		t.Error("a changed spec_draft_n_max must force a reload")
	}
	if base == (Profile{NumCtx: ptr(65536)}.Runtime("m", "/blob", "", false)) {
		t.Error("a changed num_ctx must force a reload")
	}
	if base == (Profile{AllowPartialOffload: ptr(true), NumGPU: ptr(18)}.Runtime("m", "/blob", "", false)) {
		t.Error("a changed partial-offload policy must force a reload")
	}
}

func TestProjectorFalseDropsProjector(t *testing.T) {
	rt := Profile{Projector: ptr(false)}.Runtime("m", "/blob", "/projector", false)
	if rt.ProjectorPath != "" {
		t.Errorf("projector = %q, want it suppressed", rt.ProjectorPath)
	}
}

func TestBackendDirIsBinaryDirPlusBackend(t *testing.T) {
	l := Llama{LibDir: "/usr/local/lib/ollama", Backend: "rocm_v7_2"}
	if got := l.BackendDir(); got != "/usr/local/lib/ollama/rocm_v7_2" {
		t.Errorf("BackendDir = %q", got)
	}
	if got := l.Binary(); got != "/usr/local/lib/ollama/llama-server" {
		t.Errorf("Binary = %q", got)
	}
}

// Ollama honours these and llama-server accepts every one of them, so silently
// dropping them was a divergence only a measurement would find.
func TestApplyCarriesEverySamplerOllamaHonours(t *testing.T) {
	p, err := Apply(Profile{}, map[string]any{
		"mirostat":          2.0,
		"mirostat_tau":      5.5,
		"mirostat_eta":      0.1,
		"presence_penalty":  1.5,
		"frequency_penalty": 0.5,
		"repeat_last_n":     64.0,
		"typical_p":         0.9,
		"num_keep":          24.0,
	})
	if err != nil {
		t.Fatal(err)
	}
	for name, set := range map[string]bool{
		"mirostat":          p.Mirostat != nil,
		"mirostat_tau":      p.MirostatTau != nil,
		"mirostat_eta":      p.MirostatEta != nil,
		"presence_penalty":  p.PresencePenalty != nil,
		"frequency_penalty": p.FrequencyPenalty != nil,
		"repeat_last_n":     p.RepeatLastN != nil,
		"typical_p":         p.TypicalP != nil,
		"num_keep":          p.NumKeep != nil,
	} {
		if !set {
			t.Errorf("option %q was dropped", name)
		}
	}
}

// KV cache streaming is a process-level setting like spec_type: settable from
// TOML and from a request's options, and a change has to force a reload.
func TestKVStreamArenaIsProcessLevel(t *testing.T) {
	p, err := Apply(Profile{}, map[string]any{"kv_stream_arena_mib": 2048.0})
	if err != nil {
		t.Fatal(err)
	}
	if got := deref(p.KVStreamArenaMiB, 0); got != 2048 {
		t.Errorf("kv_stream_arena_mib = %d, want 2048", got)
	}

	merged := Merge(Profile{KVStreamArenaMiB: ptr(1024)}, p)
	if got := deref(merged.KVStreamArenaMiB, 0); got != 2048 {
		t.Errorf("merged kv_stream_arena_mib = %d, want the override to win", got)
	}

	base := Profile{}.Runtime("m", "/blob", "", false)
	if base.KVStreamArenaMiB != 0 {
		t.Errorf("unset kv_stream_arena_mib resolved to %d, want streaming off", base.KVStreamArenaMiB)
	}
	streaming := merged.Runtime("m", "/blob", "", false)
	if streaming.KVStreamArenaMiB != 2048 {
		t.Errorf("Runtime dropped kv_stream_arena_mib, got %d", streaming.KVStreamArenaMiB)
	}
	if base == streaming {
		t.Error("a changed kv_stream_arena_mib must force a reload")
	}
}

// Putting expert tensors on the CPU changes what the process has to be started
// with, so both keys have to survive TOML, a request's options and a merge, and
// force a reload when they change.
func TestMoEOffloadIsProcessLevel(t *testing.T) {
	p, err := Apply(Profile{}, map[string]any{
		"num_cpu_moe":     12.0,
		"override_tensor": []any{`blk\.[0-9]+\.ffn_.*_exps=CPU`},
	})
	if err != nil {
		t.Fatal(err)
	}
	if got := deref(p.NumCPUMoE, 0); got != 12 {
		t.Errorf("num_cpu_moe = %d, want 12", got)
	}

	merged := Merge(Profile{NumCPUMoE: ptr(4), OverrideTensor: []string{"a=CPU"}}, p)
	if got := deref(merged.NumCPUMoE, 0); got != 12 {
		t.Errorf("merged num_cpu_moe = %d, want the override to win", got)
	}
	if len(merged.OverrideTensor) != 1 || merged.OverrideTensor[0] == "a=CPU" {
		t.Errorf("merged override_tensor = %v, want the override to replace the base", merged.OverrideTensor)
	}

	base := Profile{}.Runtime("m", "/blob", "", false)
	if base.NumCPUMoE != 0 || base.OverrideTensor != "" {
		t.Errorf("unset MoE offload resolved to %d/%q, want nothing placed", base.NumCPUMoE, base.OverrideTensor)
	}
	if base == merged.Runtime("m", "/blob", "", false) {
		t.Error("a changed num_cpu_moe must force a reload")
	}
	if base == (Profile{OverrideTensor: []string{"a=CPU"}}.Runtime("m", "/blob", "", false)) {
		t.Error("a changed override_tensor must force a reload")
	}
}

// The expert cache is an unmerged draft upstream, so it has to stay off unless
// asked for and reload like any other process-level setting when it is.
func TestMoEExpertCacheIsOffUnlessAskedFor(t *testing.T) {
	p, err := Apply(Profile{}, map[string]any{
		"moe_expert_cache":         8.0,
		"moe_expert_cache_inserts": 2.0,
	})
	if err != nil {
		t.Fatal(err)
	}
	merged := Merge(Profile{MoEExpertCache: ptr(4)}, p)
	rt := merged.Runtime("m", "/blob", "", false)
	if rt.MoEExpertCache != 8 || rt.MoEExpertCacheInserts != 2 {
		t.Errorf("moe_expert_cache = %d/%d, want 8/2", rt.MoEExpertCache, rt.MoEExpertCacheInserts)
	}
	base := Profile{}.Runtime("m", "/blob", "", false)
	if base.MoEExpertCache != 0 {
		t.Errorf("unset moe_expert_cache resolved to %d, want the cache disabled", base.MoEExpertCache)
	}
	if base == rt {
		t.Error("a changed moe_expert_cache must force a reload")
	}
}

// Upstream streams one sequence only, so a profile asking for both has to fail
// at load rather than at the first concurrent request.
func TestRejectsKVStreamArenaWithParallelAboveOne(t *testing.T) {
	if _, err := Load(write(t, "[defaults]\nkv_stream_arena_mib = 2048\nparallel = 4\n")); err == nil {
		t.Fatal("expected kv_stream_arena_mib with parallel = 4 to be rejected")
	}
	if _, err := Load(write(t, "[defaults]\nkv_stream_arena_mib = 2048\nparallel = 1\n")); err != nil {
		t.Fatalf("parallel = 1 is the supported case: %v", err)
	}
}

// An embedding model needs a differently-started process, so the flag has to
// reach Runtime and make two runtimes compare unequal.
func TestEmbeddingIsPartOfTheRuntimeIdentity(t *testing.T) {
	chat := Profile{}.Runtime("m", "/blob", "", false)
	embed := Profile{}.Runtime("m", "/blob", "", true)
	if chat == embed {
		t.Error("an embedding runtime compares equal to a chat one, so it would never reload")
	}
	if !embed.Embedding {
		t.Error("Runtime dropped the embedding flag")
	}
	// The profile overrides what the store detected.
	off := Profile{Embeddings: ptr(false)}.Runtime("m", "/blob", "", true)
	if off.Embedding {
		t.Error("embeddings = false did not override detection")
	}
}

// Placement flags are process-level: settable from TOML (list or comma string)
// and from a request's options, and any change has to force a reload.
func TestGPUPlacementIsProcessLevel(t *testing.T) {
	p, err := Apply(Profile{}, map[string]any{
		"device":        "Vulkan1, Vulkan0",
		"tensor_split":  "12,6",
		"split_mode":    "layer",
		"main_gpu":      1.0,
		"no_kv_offload": true,
	})
	if err != nil {
		t.Fatal(err)
	}
	rt := p.Runtime("m", "/blob", "", false)
	if rt.Device != "Vulkan1,Vulkan0" || rt.TensorSplit != "12,6" || rt.SplitMode != "layer" ||
		rt.MainGPU != 1 || !rt.NoKVOffload {
		t.Errorf("placement not carried into Runtime: %+v", rt)
	}
	list, err := Apply(Profile{}, map[string]any{"device": []any{"Vulkan1", "Vulkan0"}})
	if err != nil {
		t.Fatal(err)
	}
	if got := list.Runtime("m", "/blob", "", false).Device; got != "Vulkan1,Vulkan0" {
		t.Errorf("device list = %q, want it comma-joined", got)
	}

	base := Profile{}.Runtime("m", "/blob", "", false)
	if base.Device != "" || base.TensorSplit != "" || base.SplitMode != "" || base.MainGPU != -1 || base.NoKVOffload {
		t.Errorf("unset placement resolved to %+v, want nothing set", base)
	}
	for name, opts := range map[string]map[string]any{
		"device":        {"device": "Vulkan0"},
		"tensor_split":  {"tensor_split": "1,1"},
		"split_mode":    {"split_mode": "row"},
		"main_gpu":      {"main_gpu": 0.0},
		"no_kv_offload": {"no_kv_offload": true},
	} {
		changed, err := Apply(Profile{}, opts)
		if err != nil {
			t.Fatal(err)
		}
		if changed.Runtime("m", "/blob", "", false) == base {
			t.Errorf("a changed %s must force a reload", name)
		}
	}

	merged := Merge(Profile{Device: StringList{"Vulkan0"}, MainGPU: ptr(0)}, p)
	if got := merged.Runtime("m", "/blob", "", false); got.Device != "Vulkan1,Vulkan0" || got.MainGPU != 1 {
		t.Errorf("merge did not let the override win: %+v", got)
	}
}

func TestLoadAcceptsDeviceAsListOrString(t *testing.T) {
	for _, body := range []string{`device = ["Vulkan1", "Vulkan0"]`, `device = "Vulkan1,Vulkan0"`} {
		cfg, err := Load(write(t, "[defaults]\n"+body+"\n"))
		if err != nil {
			t.Fatalf("%s: %v", body, err)
		}
		if got := cfg.Defaults.Runtime("m", "/blob", "", false).Device; got != "Vulkan1,Vulkan0" {
			t.Errorf("%s: device = %q", body, got)
		}
	}
}

func TestStoreRootsExpandHome(t *testing.T) {
	cfg, err := Load(write(t, `
[store]
roots = ["~/models", "/var/lib/ollama/.ollama/models"]
`))
	if err != nil {
		t.Fatal(err)
	}
	home, err := os.UserHomeDir()
	if err != nil {
		t.Skip("no home directory")
	}
	want := []string{filepath.Join(home, "models"), "/var/lib/ollama/.ollama/models"}
	if len(cfg.Store.Roots) != len(want) {
		t.Fatalf("roots = %v", cfg.Store.Roots)
	}
	for i := range want {
		if cfg.Store.Roots[i] != want[i] {
			t.Errorf("roots[%d] = %q, want %q", i, cfg.Store.Roots[i], want[i])
		}
	}
}

// No [store] section leaves the choice of roots to the caller rather than
// meaning there are none.
func TestStoreRootsDefaultToNothing(t *testing.T) {
	cfg, err := Load(write(t, "[server]\nlisten = \":1\"\n"))
	if err != nil {
		t.Fatal(err)
	}
	if len(cfg.Store.Roots) != 0 {
		t.Errorf("roots = %v, want none", cfg.Store.Roots)
	}
}

func TestRejectsBadPlacementValues(t *testing.T) {
	for name, opts := range map[string]map[string]any{
		"split_mode":   {"split_mode": "diagonal"},
		"tensor_split": {"tensor_split": "12;6"},
		"main_gpu":     {"main_gpu": -1.0},
	} {
		if _, err := Apply(Profile{}, opts); err == nil {
			t.Errorf("expected %s to be rejected", name)
		}
	}
}
