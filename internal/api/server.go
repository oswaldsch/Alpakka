// Package api serves ollama's HTTP surface, backed by a supervised
// llama-server.
package api

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/api"

	"github.com/oswald/alpakka/internal/config"
	"github.com/oswald/alpakka/internal/store"
	"github.com/oswald/alpakka/internal/supervisor"
)

// Version is reported by /api/version. Clients gate features on it, so it
// claims compatibility with the ollama release alpakka's types come from.
const Version = "0.32.0"

// Server wires the store, the config and the supervisor into ollama's API.
type Server struct {
	Store  *store.Store
	Config config.Config
	Super  *supervisor.Supervisor
	Logger *log.Logger

	// loadMu serialises model loads. One card, one model: without this two
	// simultaneous requests for different models would fight over the GPU,
	// each tearing down the other's server.
	loadMu sync.Mutex
}

// Handler returns the routed HTTP handler.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()

	// Registered without a method so it also serves as the catch-all; a
	// method-scoped "/" would conflict with the more specific routes below.
	mux.HandleFunc("/", s.handleRoot)
	mux.HandleFunc("GET /api/version", s.handleVersion)
	mux.HandleFunc("GET /api/tags", s.handleTags)
	mux.HandleFunc("POST /api/show", s.handleShow)
	mux.HandleFunc("GET /api/ps", s.handlePS)
	mux.HandleFunc("POST /api/chat", s.handleChat)
	mux.HandleFunc("POST /api/generate", s.handleGenerate)
	mux.HandleFunc("POST /api/embed", s.handleEmbed)
	mux.HandleFunc("POST /api/embeddings", s.handleEmbeddings)

	// The OpenAI surface is proxied straight through to llama-server, which
	// already speaks it. Only model routing and option injection happen here.
	mux.HandleFunc("/v1/chat/completions", s.handleOpenAI)
	mux.HandleFunc("/v1/completions", s.handleOpenAI)
	mux.HandleFunc("/v1/embeddings", s.handleOpenAI)
	mux.HandleFunc("GET /v1/models", s.handleOpenAIModels)

	// Writing to the model store is out of scope. Say so plainly rather than
	// half-implementing it.
	for _, p := range []string{"/api/pull", "/api/create", "/api/push", "/api/copy", "/api/delete"} {
		mux.HandleFunc(p, s.handleUnsupported)
	}

	return s.logRequests(mux)
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
		"alpakka does not implement %s: it serves models from ollama's store read-only. "+
			"Use `ollama pull` to add a model, then request it here", r.URL.Path))
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
			// One model, fully on the GPU: that is the invariant the fit check
			// enforces, so the resident size is the whole model.
			entry.SizeVRAM = m.Size
		}
		resp.Models = append(resp.Models, entry)
	}
	writeJSON(w, http.StatusOK, resp)
}

// resolve turns a request's model name and options into a ready instance.
//
// It returns how long readiness took, which becomes load_duration. Requests
// that arrive while a model is loading block here rather than failing.
func (s *Server) resolve(ctx context.Context, name string, opts map[string]any, keepAlive *api.Duration) (
	*supervisor.Instance, config.Profile, time.Duration, error) {

	m, err := s.Store.Get(name)
	if err != nil {
		return nil, config.Profile{}, 0, errModelNotFound{name}
	}

	profile := s.Config.ForModel(m.Name)
	profile, err = config.Apply(profile, opts)
	if err != nil {
		return nil, profile, 0, errBadRequest{err}
	}

	rt := profile.Runtime(m.Name, m.ModelPath, m.ProjectorPath)

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

// writeResolveError maps a resolve failure onto the status code ollama uses.
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
	// Ollama's exact wording and status; clients match on both.
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

// formatParameters renders a params blob the way `ollama show` prints it.
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
