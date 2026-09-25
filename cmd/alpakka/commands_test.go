package main

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"reflect"
	"runtime/debug"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/types/model"

	"github.com/oswaldsch/alpakka/internal/gguf/gguftest"
	"github.com/oswaldsch/alpakka/internal/store"
)

func TestEveryCommandIsReachable(t *testing.T) {
	seen := map[string]bool{}
	for _, c := range commands() {
		for _, n := range c.names {
			if seen[n] {
				t.Errorf("%q is claimed twice", n)
			}
			seen[n] = true
		}
	}
	for _, want := range []string{"serve", "pull", "list", "ls", "ps", "show", "run", "stop", "rm", "logs", "version", "help"} {
		if !seen[want] {
			t.Errorf("no command named %q", want)
		}
	}
}

func TestOptionFlagsTypeWhatParses(t *testing.T) {
	o := optionFlags{}
	for _, s := range []string{"reasoning_effort=low", "num_ctx=65536", "flash_attn=true", `stop=["</s>"]`, "seed="} {
		if err := o.Set(s); err != nil {
			t.Fatalf("Set(%q): %v", s, err)
		}
	}
	want := optionFlags{
		"reasoning_effort": "low",
		"num_ctx":          float64(65536),
		"flash_attn":       true,
		"stop":             []any{"</s>"},
		"seed":             "",
	}
	if !reflect.DeepEqual(o, want) {
		t.Errorf("got %v, want %v", o, want)
	}
	for _, bad := range []string{"novalue", "=x"} {
		if err := o.Set(bad); err == nil {
			t.Errorf("Set(%q) was accepted", bad)
		}
	}
}

func fakeServer(t *testing.T, handler http.HandlerFunc) string {
	t.Helper()
	srv := httptest.NewServer(handler)
	t.Cleanup(srv.Close)
	return strings.TrimPrefix(srv.URL, "http://")
}

func TestChatTurnStreamsAndKeepsHistory(t *testing.T) {
	var got []api.ChatRequest
	host := fakeServer(t, func(w http.ResponseWriter, r *http.Request) {
		var req api.ChatRequest
		json.NewDecoder(r.Body).Decode(&req)
		got = append(got, req)
		enc := json.NewEncoder(w)
		enc.Encode(api.ChatResponse{Message: api.Message{Thinking: "hmm"}})
		enc.Encode(api.ChatResponse{Message: api.Message{Content: "hel"}})
		enc.Encode(api.ChatResponse{Message: api.Message{Content: "lo"}})
		enc.Encode(api.ChatResponse{Done: true, Metrics: api.Metrics{EvalCount: 30, EvalDuration: time.Second}})
	})

	var out, thinking, meta bytes.Buffer
	c := &chatSession{
		client:   api.NewClient(&url.URL{Scheme: "http", Host: host}, http.DefaultClient),
		model:    "m",
		options:  map[string]any{"reasoning_effort": "low"},
		history:  []api.Message{{Role: "system", Content: "be brief"}},
		out:      &out,
		thinking: &thinking,
		meta:     &meta,
		verbose:  true,
	}
	for _, prompt := range []string{"hi", "again"} {
		if err := c.turn(t.Context(), prompt); err != nil {
			t.Fatal(err)
		}
	}

	if out.String() != "hello\nhello\n" {
		t.Errorf("stdout = %q, want only the answers", out.String())
	}
	if !strings.HasPrefix(thinking.String(), "hmm\n\n") {
		t.Errorf("thinking = %q", thinking.String())
	}
	if !strings.Contains(meta.String(), "decode 30 tokens, 30.00 tokens/s") {
		t.Errorf("stats = %q", meta.String())
	}
	if len(got) != 2 || got[1].Options["reasoning_effort"] != "low" {
		t.Fatalf("requests = %+v", got)
	}
	roles := []string{}
	for _, m := range got[1].Messages {
		roles = append(roles, m.Role+":"+m.Content)
	}
	if want := []string{"system:be brief", "user:hi", "assistant:hello", "user:again"}; !reflect.DeepEqual(roles, want) {
		t.Errorf("second request carried %v, want %v", roles, want)
	}
	if c.history = systemOnly(c.history); len(c.history) != 1 {
		t.Errorf("/clear kept %d messages, want the system prompt alone", len(c.history))
	}
}

func TestChatTurnFailureLeavesHistoryAlone(t *testing.T) {
	host := fakeServer(t, func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, `{"error":"model 'm' not found"}`, http.StatusNotFound)
	})
	c := &chatSession{
		client: api.NewClient(&url.URL{Scheme: "http", Host: host}, http.DefaultClient),
		model:  "m",
		out:    &bytes.Buffer{},
		meta:   &bytes.Buffer{},
	}
	if err := c.turn(t.Context(), "hi"); err == nil || !strings.Contains(err.Error(), "not found") {
		t.Fatalf("err = %v, want the server's message", err)
	}
	if len(c.history) != 0 {
		t.Errorf("history = %v after a failed turn", c.history)
	}
}

func TestStopPostsTheModelAndReportsTheServerError(t *testing.T) {
	var asked struct {
		Model string
		Force bool
	}
	host := fakeServer(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/ps" {
			w.Write([]byte(`{"models":[]}`))
			return
		}
		json.NewDecoder(r.Body).Decode(&asked)
		if r.URL.Path != "/alpakka/unload" || asked.Model == "other" {
			w.WriteHeader(http.StatusNotFound)
			w.Write([]byte(`{"error":"model 'other' is not loaded"}`))
			return
		}
		w.Write([]byte(`{"model":"m:1"}`))
	})
	if err := run([]string{"stop", "-host", host, "m"}); err != nil || asked.Model != "m" || asked.Force {
		t.Fatalf("stop m: err %v, asked for %+v", err, asked)
	}
	if err := run([]string{"stop", "-host", host, "--force"}); err != nil || !asked.Force {
		t.Fatalf("stop --force: err %v, asked for %+v", err, asked)
	}
	err := run([]string{"stop", "-host", host, "other"})
	if err == nil || err.Error() != "model 'other' is not loaded" {
		t.Errorf("err = %v, want the server's message alone", err)
	}
}

func TestStopNoticesAResponseInProgress(t *testing.T) {
	var resp psResponse
	resp.Models = make([]psEntry, 1)
	resp.Models[0].Name = "qwen3.8-27b:q3-k-xl"

	if _, busy := generating(resp, ""); busy {
		t.Error("an idle model was reported generating")
	}
	resp.Models[0].Alpakka.Busy = true
	for _, name := range []string{"", "qwen3.8-27b", "qwen3.8-27b:q3-k-xl"} {
		if got, busy := generating(resp, name); !busy || got != "qwen3.8-27b:q3-k-xl" {
			t.Errorf("generating(%q) = %q, %t", name, got, busy)
		}
	}
	if _, busy := generating(resp, "other"); busy {
		t.Error("a different model's response was waited on")
	}
}

func TestPrintLogsKeepsTheHeaderOffStdout(t *testing.T) {
	now := time.Date(2026, 9, 23, 12, 0, 0, 0, time.UTC)
	var out, meta bytes.Buffer
	printLogs(&out, &meta, logsResponse{
		Model: "m:1", StartedAt: now.Add(-2 * time.Hour), Lines: []string{"a", "b"},
	}, now)
	if out.String() != "a\nb\n" {
		t.Errorf("stdout = %q", out.String())
	}
	if !strings.Contains(meta.String(), "m:1, exited, started 2 hours ago") {
		t.Errorf("header = %q", meta.String())
	}

	out.Reset()
	meta.Reset()
	printLogs(&out, &meta, logsResponse{}, now)
	if out.Len() != 0 || !strings.Contains(meta.String(), "no llama-server") {
		t.Errorf("empty log: stdout %q, stderr %q", out.String(), meta.String())
	}
}

func TestPrintShow(t *testing.T) {
	var out bytes.Buffer
	printShow(&out, "qwen3:0.6b", api.ShowResponse{
		Details:      api.ModelDetails{Family: "qwen3", QuantizationLevel: "Q4_K_M", ContextLength: 40960},
		ModelInfo:    map[string]any{"qwen3.block_count": float64(28)},
		Capabilities: []model.Capability{"completion", "tools"},
	})
	for _, want := range []string{"qwen3:0.6b", "Q4_K_M", "40960", "layers", "28", "tools"} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("show output missing %q:\n%s", want, out.String())
		}
	}
	if strings.Contains(out.String(), "embedding length") {
		t.Errorf("an unset field was printed:\n%s", out.String())
	}
}

func writeModel(t *testing.T, path string) {
	t.Helper()
	gguftest.Write(t, path, map[string]any{"general.architecture": "qwen3", "general.file_type": uint32(15)})
}

func TestRemoveModelKeepsASharedProjectorForTheOtherTags(t *testing.T) {
	root := t.TempDir()
	dir := filepath.Join(root, "vl")
	for _, f := range []string{"q4-k-m.gguf", "q8-0.gguf", "mmproj-F16.gguf"} {
		writeModel(t, filepath.Join(dir, f))
	}
	s := store.New(nil, root)

	m, err := s.Get("vl:q4-k-m")
	if err != nil {
		t.Fatal(err)
	}
	if err := removeModel(m); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(dir, "mmproj-F16.gguf")); err != nil {
		t.Fatal("the projector went while another tag still uses it")
	}

	m, err = s.Get("vl:q8-0")
	if err != nil {
		t.Fatal(err)
	}
	if err := removeModel(m); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(dir); !os.IsNotExist(err) {
		t.Errorf("the emptied model directory is still there: %v", err)
	}
}

func TestRemoveModelTakesEveryPartAndLeavesStrangersAlone(t *testing.T) {
	root := t.TempDir()
	dir := filepath.Join(root, "big")
	writeModel(t, filepath.Join(dir, "q4-k-m-00001-of-00002.gguf"))
	writeModel(t, filepath.Join(dir, "q4-k-m-00002-of-00002.gguf"))
	writeModel(t, filepath.Join(dir, "q4-k-m.mmproj.gguf"))
	if err := os.WriteFile(filepath.Join(dir, "q8-0.gguf.part"), []byte("half"), 0o644); err != nil {
		t.Fatal(err)
	}

	m, err := store.New(nil, root).Get("big:q4-k-m")
	if err != nil {
		t.Fatal(err)
	}
	if err := removeModel(m); err != nil {
		t.Fatal(err)
	}
	entries, _ := os.ReadDir(dir)
	if len(entries) != 1 || entries[0].Name() != "q8-0.gguf.part" {
		t.Errorf("left %v, want only the unrelated partial download", entries)
	}
}

func TestLlamaFeatures(t *testing.T) {
	help := "--fit --spec-type --jinja --no-webui --no-mmproj\n-fa, --flash-attn [on|off|auto]\n--n-cpu-moe N"
	var out bytes.Buffer
	if err := printFeatures(&out, llamaFeatures(help)); err != nil {
		t.Fatalf("a build with every required flag failed: %v", err)
	}
	if !strings.Contains(out.String(), "moe_expert_cache unavailable") {
		t.Errorf("an absent optional flag was not reported:\n%s", out.String())
	}

	old := "--fit --jinja --no-webui --no-mmproj\n-fa, --flash-attn"
	err := printFeatures(&bytes.Buffer{}, llamaFeatures(old))
	if err == nil || !strings.Contains(err.Error(), "--spec-type") || !strings.Contains(err.Error(), "--flash-attn") {
		t.Errorf("err = %v, want it to name --spec-type and the boolean --flash-attn", err)
	}
}

func TestBuildVersion(t *testing.T) {
	info := &debug.BuildInfo{
		Main:     debug.Module{Version: "(devel)"},
		Settings: []debug.BuildSetting{{Key: "vcs.revision", Value: "c4db128d36aa0123"}, {Key: "vcs.modified", Value: "true"}},
	}
	if got := buildVersion(info, true); got != "(devel) (c4db128d36aa, modified)" {
		t.Errorf("got %q", got)
	}
	info.Main.Version = "v0.0.0-20260924150351-c4db128d36aa+dirty"
	if got := buildVersion(info, true); got != info.Main.Version {
		t.Errorf("got %q, want the pseudo-version alone", got)
	}
	if got := buildVersion(nil, false); got != "unknown" {
		t.Errorf("got %q", got)
	}
}

func TestJoinPromptPutsPipedTextAfterTheInstruction(t *testing.T) {
	cases := []struct{ arg, stdin, want string }{
		{"review this", "diff --git a b\n", "review this\n\ndiff --git a b"},
		{"", "  just stdin\n", "just stdin"},
		{"just the arg", "\n", "just the arg"},
		{"", "", ""},
	}
	for _, c := range cases {
		if got := joinPrompt(c.arg, c.stdin); got != c.want {
			t.Errorf("joinPrompt(%q, %q) = %q, want %q", c.arg, c.stdin, got, c.want)
		}
	}
}

func TestChatTurnReportsACutOffAnswer(t *testing.T) {
	host := fakeServer(t, func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(api.ChatResponse{Message: api.Message{Content: "you "}})
	})
	c := &chatSession{
		client: api.NewClient(&url.URL{Scheme: "http", Host: host}, http.DefaultClient),
		model:  "m",
		out:    &bytes.Buffer{},
		meta:   &bytes.Buffer{},
	}
	if err := c.turn(t.Context(), "hi"); err == nil || !strings.Contains(err.Error(), "cut off") {
		t.Fatalf("err = %v, want the answer reported as cut off", err)
	}
	if len(c.history) != 0 {
		t.Errorf("a cut-off answer went into the history: %v", c.history)
	}
}
