package api

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	ollama "github.com/ollama/ollama/api"

	"github.com/oswald/alpakka/internal/config"
	"github.com/oswald/alpakka/internal/store"
	"github.com/oswald/alpakka/internal/supervisor"
)

// testServer builds a server over the real model store. Only endpoints that do
// not need a loaded model are exercised here; the supervisor is covered by its
// own tests against a real llama-server.
func testServer(t *testing.T) http.Handler {
	t.Helper()
	root := store.DefaultRoot()
	if _, err := os.Stat(root); err != nil {
		t.Skip("ollama model store not present")
	}
	cfg := config.Default()
	s := &Server{
		Store:  store.New(root),
		Config: cfg,
		Super:  supervisor.New(cfg.Llama, nil),
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

// Go's mux panics on conflicting route patterns, so building the handler at all
// is a real check.
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
	// Ollama's exact wording; clients match on it.
	if e["error"] != "model 'definitely-absent' not found" {
		t.Errorf("error = %q", e["error"])
	}
}

// /api/ps must be a well-formed empty list when nothing is loaded, not null.
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

// A bad option must be rejected before a slow model load, not after.
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

func TestFormatParametersMatchesOllamaLayout(t *testing.T) {
	got := formatParameters(map[string]any{
		"num_ctx": float64(32768),
		"stop":    []any{"<|im_start|>", "<|im_end|>"},
	})
	want := "num_ctx                        32768\n" +
		"stop                           \"<|im_start|>\"\n" +
		"stop                           \"<|im_end|>\""
	if got != want {
		t.Errorf("parameters =\n%q\nwant\n%q", got, want)
	}
}
