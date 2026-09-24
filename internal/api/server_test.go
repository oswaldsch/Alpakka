package api

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"

	"github.com/oswald/alpakka/internal/gguf/gguftest"

	ollama "github.com/ollama/ollama/api"

	"github.com/oswald/alpakka/internal/config"
	"github.com/oswald/alpakka/internal/store"
	"github.com/oswald/alpakka/internal/supervisor"
)

// Only endpoints that need no loaded model are exercised, the supervisor has its own tests.
func testServer(t *testing.T) http.Handler {
	t.Helper()
	root := t.TempDir()
	gguftest.Write(t, filepath.Join(root, "qwen3", "0.6b.gguf"), map[string]any{
		"general.architecture":    "qwen3",
		"general.file_type":       uint32(15),
		"qwen3.context_length":    uint32(40960),
		"tokenizer.chat_template": "{% for m in messages %}{{ m.content }}{% endfor %}",
	})
	cfg := config.Default()
	s := &Server{
		Store:  store.New(nil, root),
		Config: cfg,
		Super:  supervisor.New(cfg.Llama, cfg.WoL, nil),
	}
	return s.Handler()
}

func do(t *testing.T, h http.Handler, method, path, body string) *httptest.ResponseRecorder {
	t.Helper()
	var r *http.Request
	if body == "" {
		r = httptest.NewRequest(method, path, nil)
	} else {
		r = httptest.NewRequest(method, path, strings.NewReader(body))
	}
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)
	return w
}

// Go's mux panics on conflicting route patterns, so building the handler at all is a check.
func TestHandlerRoutesDoNotConflict(t *testing.T) {
	if testServer(t) == nil {
		t.Fatal("nil handler")
	}
}

// Clients probe the root for this exact string to decide ollama is up.
func TestRootIdentifiesAsOllama(t *testing.T) {
	w := do(t, testServer(t), http.MethodGet, "/", "")
	if got := w.Body.String(); got != "Ollama is running" {
		t.Errorf("root = %q", got)
	}
}

func TestVersion(t *testing.T) {
	w := do(t, testServer(t), http.MethodGet, "/api/version", "")
	var v map[string]string
	if err := json.Unmarshal(w.Body.Bytes(), &v); err != nil {
		t.Fatal(err)
	}
	if v["version"] == "" {
		t.Error("no version reported")
	}
}

func TestTagsListsTheStore(t *testing.T) {
	w := do(t, testServer(t), http.MethodGet, "/api/tags", "")
	if w.Code != http.StatusOK {
		t.Fatalf("status %d", w.Code)
	}
	var resp ollama.ListResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if len(resp.Models) == 0 {
		t.Fatal("no models listed")
	}
	for _, m := range resp.Models {
		if m.Name == "" || m.Digest == "" || m.Size == 0 {
			t.Errorf("incomplete entry: %+v", m)
		}
	}
}

func TestShowUnknownModelMatchesOllama(t *testing.T) {
	w := do(t, testServer(t), http.MethodPost, "/api/show", `{"model":"definitely-absent"}`)
	if w.Code != http.StatusNotFound {
		t.Errorf("status = %d, want 404", w.Code)
	}
	var e map[string]string
	_ = json.Unmarshal(w.Body.Bytes(), &e)
	// Ollama's exact wording, since clients match on it.
	if e["error"] != "model 'definitely-absent' not found" {
		t.Errorf("error = %q", e["error"])
	}
}

// Must be a well-formed empty list when nothing is loaded, not null.
func TestPSEmptyWhenNothingLoaded(t *testing.T) {
	w := do(t, testServer(t), http.MethodGet, "/api/ps", "")
	if !strings.Contains(w.Body.String(), `"models":[]`) {
		t.Errorf("ps = %s", w.Body.String())
	}
}

func TestWriteEndpointsAreRefusedClearly(t *testing.T) {
	h := testServer(t)
	for _, p := range []string{"/api/pull", "/api/create", "/api/push", "/api/copy", "/api/delete"} {
		w := do(t, h, http.MethodPost, p, `{"model":"x"}`)
		if w.Code != http.StatusNotImplemented {
			t.Errorf("%s: status = %d, want 501", p, w.Code)
		}
		if !strings.Contains(w.Body.String(), "ollama pull") {
			t.Errorf("%s: error should point at ollama pull, got %s", p, w.Body.String())
		}
	}
}

func TestInvalidReasoningEffortIsRejected(t *testing.T) {
	w := do(t, testServer(t), http.MethodPost, "/api/chat",
		`{"model":"qwen3:0.6b","messages":[],"options":{"reasoning_effort":"high"}}`)
	if w.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400 (body: %s)", w.Code, w.Body.String())
	}
	if !strings.Contains(w.Body.String(), "xhigh") {
		t.Errorf("error should name the accepted values, got %s", w.Body.String())
	}
}

func TestUnknownPathIs404(t *testing.T) {
	w := do(t, testServer(t), http.MethodGet, "/api/nonsense", "")
	if w.Code != http.StatusNotFound {
		t.Errorf("status = %d, want 404", w.Code)
	}
}

func TestOpenAIModelsListsTheStore(t *testing.T) {
	w := do(t, testServer(t), http.MethodGet, "/v1/models", "")
	var resp struct {
		Object string           `json:"object"`
		Data   []map[string]any `json:"data"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if resp.Object != "list" || len(resp.Data) == 0 {
		t.Errorf("v1/models = %s", w.Body.String())
	}
}

func TestPreflightIsAnswered(t *testing.T) {
	h := testServer(t)
	r := httptest.NewRequest(http.MethodOptions, "/api/chat", nil)
	r.Header.Set("Origin", "http://localhost:3000")
	r.Header.Set("Access-Control-Request-Method", "POST")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)

	if w.Code != http.StatusNoContent {
		t.Errorf("preflight status = %d, want 204", w.Code)
	}
	if got := w.Header().Get("Access-Control-Allow-Origin"); got != "http://localhost:3000" {
		t.Errorf("Allow-Origin = %q", got)
	}
	if !strings.Contains(w.Header().Get("Access-Control-Allow-Methods"), "POST") {
		t.Errorf("Allow-Methods = %q", w.Header().Get("Access-Control-Allow-Methods"))
	}
}

func TestPreflightAllowsAnthropicHeaders(t *testing.T) {
	h := testServer(t)
	r := httptest.NewRequest(http.MethodOptions, "/v1/messages", nil)
	r.Header.Set("Origin", "http://localhost:3000")
	r.Header.Set("Access-Control-Request-Method", "POST")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)

	allowed := w.Header().Get("Access-Control-Allow-Headers")
	for _, hd := range []string{
		"anthropic-version", "anthropic-beta", "x-api-key",
		"anthropic-dangerous-direct-browser-access",
	} {
		if !strings.Contains(allowed, hd) {
			t.Errorf("Allow-Headers %q does not include %s", allowed, hd)
		}
	}
}

func TestCORSAllowsLocalAndAppOriginsOnly(t *testing.T) {
	s := &Server{Config: config.Default()}
	allowed := []string{
		"http://localhost:8080", "http://127.0.0.1:3000", "https://localhost",
		"app://obsidian.md", "chrome-extension://abcdef", "vscode-webview://x",
	}
	for _, o := range allowed {
		if !s.originAllowed(o) {
			t.Errorf("%s should be allowed", o)
		}
	}
	for _, o := range []string{"https://evil.example", "http://192.168.1.9", ""} {
		if s.originAllowed(o) {
			t.Errorf("%s should not be allowed", o)
		}
	}

	s.Config.Server.Origins = []string{"https://chat.example"}
	if s.originAllowed("http://localhost:3000") {
		t.Error("a configured list should not still allow localhost")
	}
	if !s.originAllowed("https://chat.example") {
		t.Error("the configured origin was rejected")
	}
}

func TestCORSHeadersOnRealResponses(t *testing.T) {
	h := testServer(t)
	r := httptest.NewRequest(http.MethodGet, "/api/version", nil)
	r.Header.Set("Origin", "http://localhost:3000")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)

	if got := w.Header().Get("Access-Control-Allow-Origin"); got != "http://localhost:3000" {
		t.Errorf("Allow-Origin = %q", got)
	}
	if !strings.Contains(w.Header().Get("Vary"), "Origin") {
		t.Errorf("Vary = %q, want it to include Origin", w.Header().Get("Vary"))
	}
}
