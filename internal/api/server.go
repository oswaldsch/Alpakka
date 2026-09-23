// Package api serves ollama's HTTP surface, backed by a supervised llama-server.
package api

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/api"

	"github.com/oswald/alpakka/internal/config"
	"github.com/oswald/alpakka/internal/store"
	"github.com/oswald/alpakka/internal/supervisor"
)

// Clients gate features on it, so it claims compatibility with the ollama release the types come from.
const Version = "0.32.0"

type Server struct {
	Store  store.Source
	Config config.Config
	Super  *supervisor.Supervisor
	Logger *log.Logger

	// One card, one model: without this two simultaneous loads of different models would
	// tear down each other's server.
	loadMu sync.Mutex
}

func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()

	// Registered without a method so it also serves as the catch-all, since a method-scoped "/"
	// would conflict with the routes below.
	mux.HandleFunc("/", s.handleRoot)
	mux.HandleFunc("GET /api/version", s.handleVersion)
	mux.HandleFunc("GET /api/tags", s.handleTags)
	mux.HandleFunc("POST /api/show", s.handleShow)
	mux.HandleFunc("GET /api/ps", s.handlePS)
	mux.HandleFunc("POST /api/chat", s.handleChat)
	mux.HandleFunc("POST /api/generate", s.handleGenerate)
	mux.HandleFunc("POST /api/embed", s.handleEmbed)
	mux.HandleFunc("POST /api/embeddings", s.handleEmbeddings)

	mux.HandleFunc("/v1/chat/completions", s.handleOpenAI)
	mux.HandleFunc("/v1/completions", s.handleOpenAI)
	mux.HandleFunc("/v1/embeddings", s.handleOpenAI)
	mux.HandleFunc("/v1/messages", s.handleOpenAI)
	mux.HandleFunc("/v1/messages/count_tokens", s.handleOpenAI)
	mux.HandleFunc("GET /v1/models", s.handleOpenAIModels)

	// alpakka's own surface, outside both wire protocols so no ollama or OpenAI client reaches it by accident.
	mux.HandleFunc("POST /alpakka/bench", s.handleBench)
	mux.HandleFunc("GET /alpakka/status", s.handleBenchStatus)

	// Writing to the model store is out of scope, so say so rather than half-implement it.
	for _, p := range []string{"/api/pull", "/api/create", "/api/push", "/api/copy", "/api/delete"} {
		mux.HandleFunc(p, s.handleUnsupported)
	}

	return s.logRequests(s.withCORS(mux))
}

var localHosts = []string{"localhost", "127.0.0.1", "0.0.0.0", "::1"}

// Desktop and extension clients present these non-http(s) origins, so there is no host to match on.
var appSchemes = []string{"app", "file", "tauri", "vscode-webview",
	"vscode-file", "moz-extension", "chrome-extension", "safari-web-extension"}

// With no configured list this mirrors ollama's default of any loopback port plus desktop
// schemes. A configured list replaces that, and "*" allows everything.
func (s *Server) originAllowed(origin string) bool {
	if origin == "" {
		return false
	}
	if configured := s.Config.Server.Origins; len(configured) > 0 {
		for _, o := range configured {
			if o == "*" || strings.EqualFold(o, origin) {
				return true
			}
		}
		return false
	}

	u, err := url.Parse(origin)
	if err != nil {
		return false
	}
	for _, scheme := range appSchemes {
		if strings.EqualFold(u.Scheme, scheme) {
			return true
		}
	}
	host := u.Hostname()
	for _, h := range localHosts {
		if strings.EqualFold(host, h) {
			return true
		}
	}
	return false
}

// The mux answers an OPTIONS preflight with 405, so without this browser clients see an
// opaque CORS failure with nothing in the server log.
func (s *Server) withCORS(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		h := w.Header()
		// Set unconditionally since the response varies by origin whether or not this one was allowed.
		h.Add("Vary", "Origin")

		if origin := r.Header.Get("Origin"); s.originAllowed(origin) {
			h.Set("Access-Control-Allow-Origin", origin)
			h.Set("Access-Control-Allow-Credentials", "true")
			h.Set("Access-Control-Allow-Methods", "GET, POST, HEAD, OPTIONS")
			h.Set("Access-Control-Allow-Headers",
				"Content-Type, Authorization, Accept, User-Agent, X-Requested-With, X-Stainless-Lang")
			h.Set("Access-Control-Max-Age", "86400")
		}

		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		next.ServeHTTP(w, r)
	})
}

func (s *Server) logf(format string, args ...any) {
	if s.Logger != nil {
		s.Logger.Printf(format, args...)
	}
}

func (s *Server) logRequests(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		next.ServeHTTP(w, r)
		if r.URL.Path != "/api/ps" && r.URL.Path != "/api/tags" {
			s.logf("%s %s (%s)", r.Method, r.URL.Path, time.Since(start).Round(time.Millisecond))
		}
	})
}

func (s *Server) handleRoot(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		writeError(w, http.StatusNotFound, "not found: "+r.URL.Path)
		return
	}
	if r.Method != http.MethodGet && r.Method != http.MethodHead {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	// Clients probe for this exact string to decide whether ollama is up.
	fmt.Fprint(w, "Ollama is running")
}

func (s *Server) handleVersion(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"version": Version})
}

func (s *Server) handleUnsupported(w http.ResponseWriter, r *http.Request) {
	writeError(w, http.StatusNotImplemented, fmt.Sprintf(
		"alpakka does not implement %s: its model store is read-only over HTTP. "+
			"Use `alpakka pull` or `ollama pull` to add a model, then request it here",
		r.URL.Path))
}

func (s *Server) handleTags(w http.ResponseWriter, r *http.Request) {
	models, err := s.Store.List()
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	resp := api.ListResponse{Models: make([]api.ListModelResponse, 0, len(models))}
	for i := range models {
		resp.Models = append(resp.Models, models[i].ListResponse())
	}
	writeJSON(w, http.StatusOK, resp)
}

func (s *Server) handleShow(w http.ResponseWriter, r *http.Request) {
	var req api.ShowRequest
	if !decode(w, r, &req) {
		return
	}
	name := req.Model
	if name == "" {
		name = req.Name
	}
	m, err := s.Store.Get(name)
	if err != nil {
		writeModelNotFound(w, name)
		return
	}

	resp := api.ShowResponse{
		License:      m.License,
		Template:     m.Template,
		System:       m.System,
		Parameters:   formatParameters(m.Params),
		Details:      m.Details(),
		Capabilities: m.Capabilities(),
		ModifiedAt:   m.ModifiedAt,
		ModelInfo:    map[string]any{},
	}
	resp.Modelfile = modelfile(m)
	if f, err := m.GGUF(); err == nil {
		resp.ModelInfo = modelInfo(f)
	}
	writeJSON(w, http.StatusOK, resp)
}

func (s *Server) handlePS(w http.ResponseWriter, r *http.Request) {
	resp := api.ProcessResponse{Models: []api.ProcessModelResponse{}}

	if inst := s.Super.Current(); inst != nil {
		rt := inst.Runtime()
		entry := api.ProcessModelResponse{
			Name:          rt.Model,
			Model:         rt.Model,
			ExpiresAt:     inst.ExpiresAt(),
			ContextLength: rt.NumCtx,
		}
		if m, err := s.Store.Get(rt.Model); err == nil {
			entry.Digest = m.Digest
			entry.Size = m.Size
			entry.Details = m.Details()
			// One model fully on the GPU is the fit check's invariant, so resident size is the weights plus
			// the KV cache llama.cpp reported. Weights alone understate VRAM by GBs at 32k and q8_0.
			entry.SizeVRAM = m.Size + int64(inst.Fit().KVBufferMiB*1024*1024)
		}
		resp.Models = append(resp.Models, entry)
	}
	writeJSON(w, http.StatusOK, resp)
}

// The returned instance carries a reference, so every caller must defer Release. forEmbedding is
// set by the endpoint since ollama embeds with any model, and a chat model asked to embed reloads into an embedding process.
func (s *Server) resolve(ctx context.Context, name string, opts map[string]any, keepAlive *api.Duration,
	forEmbedding, reuseResident bool) (*supervisor.Instance, config.Profile, time.Duration, error) {

	m, err := s.Store.Get(name)
	if err != nil {
		return nil, config.Profile{}, 0, errModelNotFound{name}
	}

	profile := s.Config.ForModel(m.Name)
	profile, err = config.Apply(profile, opts)
	if err != nil {
		return nil, profile, 0, errBadRequest{err}
	}

	rt := profile.Runtime(m.Name, m.ModelPath, m.ProjectorPath, forEmbedding || m.IsEmbedding())
	if rt.Embedding && rt.Pooling == "" && !m.DeclaresPooling() {
		// A causal model has no pooling type and llama.cpp's OpenAI endpoint rejects "none", so pick the
		// reduction decoder models are trained for rather than fail.
		rt.Pooling = "last"
	}
	if rt.Embedding {
		// An embedding model has no context to extend, so asking for more than it was trained on is a load
		// that --fit off refuses. The chat-oriented global num_ctx default would fail every embedding load.
		if trained := m.TrainedContext(); trained > 0 && rt.NumCtx > trained {
			s.logf("%s: num_ctx %d exceeds the model's trained %d, using %d",
				m.Name, rt.NumCtx, trained, trained)
			rt.NumCtx = trained
		}
	}

	// Tokenizing needs only the right vocab, so a runtime-setting mismatch is no reason to reload.
	if reuseResident {
		if inst := s.Super.AcquireIfModel(ctx, rt.Model, rt.Embedding); inst != nil {
			return inst, profile, 0, nil
		}
	}

	s.loadMu.Lock()
	defer s.loadMu.Unlock()

	start := time.Now()
	inst, err := s.Super.Ensure(ctx, rt)
	if err != nil {
		return nil, profile, 0, err
	}
	load := time.Since(start)

	keep := profile.KeepAliveDuration()
	if keepAlive != nil {
		keep = keepAlive.Duration
	}
	inst.Touch(keep)

	return inst, profile, load, nil
}

type errModelNotFound struct{ name string }

func (e errModelNotFound) Error() string {
	return fmt.Sprintf("model %q not found", e.name)
}

type errBadRequest struct{ err error }

func (e errBadRequest) Error() string { return e.err.Error() }

func writeResolveError(w http.ResponseWriter, name string, err error) {
	var nf errModelNotFound
	var bad errBadRequest
	switch {
	case errorsAs(err, &nf):
		writeModelNotFound(w, name)
	case errorsAs(err, &bad):
		writeError(w, http.StatusBadRequest, bad.Error())
	default:
		writeError(w, http.StatusInternalServerError, err.Error())
	}
}

func writeModelNotFound(w http.ResponseWriter, name string) {
	// Ollama's exact wording and status, since clients match on both.
	writeError(w, http.StatusNotFound, fmt.Sprintf("model '%s' not found", name))
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]string{"error": msg})
}

func decode(w http.ResponseWriter, r *http.Request, v any) bool {
	if err := json.NewDecoder(r.Body).Decode(v); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body: "+err.Error())
		return false
	}
	return true
}

func formatParameters(params map[string]any) string {
	if len(params) == 0 {
		return ""
	}
	var b strings.Builder
	keys := sortedKeys(params)
	for _, k := range keys {
		switch v := params[k].(type) {
		case []any:
			for _, e := range v {
				fmt.Fprintf(&b, "%-30s %s\n", k, quoteParam(e))
			}
		default:
			fmt.Fprintf(&b, "%-30s %s\n", k, quoteParam(v))
		}
	}
	return strings.TrimRight(b.String(), "\n")
}

func quoteParam(v any) string {
	if s, ok := v.(string); ok {
		return fmt.Sprintf("%q", s)
	}
	if f, ok := v.(float64); ok && f == float64(int64(f)) {
		return fmt.Sprintf("%d", int64(f))
	}
	return fmt.Sprint(v)
}
