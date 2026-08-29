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
	if _, err := Load(write(t, "[defaults]\nreasoning_effort = \"high\"\n")); err == nil {
		t.Fatal("expected reasoning_effort = high to be rejected")
	}
}

func TestRejectsBadKeepAlive(t *testing.T) {
	if _, err := Load(write(t, "[defaults]\nkeep_alive = \"soon\"\n")); err == nil {
		t.Fatal("expected an invalid keep_alive to be rejected")
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
	base := Profile{}.Runtime("m", "/blob", "")
	if base != (Profile{}.Runtime("m", "/blob", "")) {
		t.Fatal("identical profiles produced unequal runtimes")
	}
	if base == (Profile{SpecDraftNMax: ptr(2)}.Runtime("m", "/blob", "")) {
		t.Error("a changed spec_draft_n_max must force a reload")
	}
	if base == (Profile{NumCtx: ptr(65536)}.Runtime("m", "/blob", "")) {
		t.Error("a changed num_ctx must force a reload")
	}
}

func TestProjectorFalseDropsProjector(t *testing.T) {
	rt := Profile{Projector: ptr(false)}.Runtime("m", "/blob", "/projector")
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
