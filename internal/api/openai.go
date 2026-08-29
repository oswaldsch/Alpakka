package api

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"time"

	"github.com/ollama/ollama/api"

	"github.com/oswald/alpakka/internal/config"
)

// handleOpenAI proxies the /v1 surface to llama-server, which already speaks
// it. alpakka's only jobs here are routing to the right model and folding in
// the configured defaults, so a client that never heard of alpakka still gets
// the speculative decoding and reasoning effort settings.
func (s *Server) handleOpenAI(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}

	raw, err := io.ReadAll(io.LimitReader(r.Body, 256<<20))
	if err != nil {
		writeError(w, http.StatusBadRequest, "reading request body: "+err.Error())
		return
	}

	var body map[string]any
	if err := json.Unmarshal(raw, &body); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body: "+err.Error())
		return
	}
	name, _ := body["model"].(string)
	if name == "" {
		writeError(w, http.StatusBadRequest, "model is required")
		return
	}

	inst, profile, _, err := s.resolve(r.Context(), name, optionsFrom(body), keepAliveFrom(body))
	if err != nil {
		writeOpenAIResolveError(w, name, err)
		return
	}

	// llama-server knows the model by the alias it was started with.
	body["model"] = inst.Runtime().Model
	applyProfileToOpenAI(body, profile)
	delete(body, "keep_alive")

	patched, err := json.Marshal(body)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}

	req, err := http.NewRequestWithContext(r.Context(), http.MethodPost,
		inst.BaseURL()+r.URL.Path, bytes.NewReader(patched))
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	req.Header.Set("Content-Type", "application/json")
	if accept := r.Header.Get("Accept"); accept != "" {
		req.Header.Set("Accept", accept)
	}

	resp, err := upstream.Do(req)
	if err != nil {
		writeError(w, http.StatusBadGateway, err.Error())
		return
	}
	defer resp.Body.Close()

	for k, vs := range resp.Header {
		for _, v := range vs {
			w.Header().Add(k, v)
		}
	}
	w.WriteHeader(resp.StatusCode)
	streamCopy(w, resp.Body)
}

// streamCopy forwards the upstream body, flushing so server-sent events reach
// the client as they arrive rather than in one lump at the end.
func streamCopy(w http.ResponseWriter, r io.Reader) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		_, _ = io.Copy(w, r)
		return
	}
	buf := make([]byte, 16*1024)
	for {
		n, err := r.Read(buf)
		if n > 0 {
			if _, werr := w.Write(buf[:n]); werr != nil {
				return
			}
			flusher.Flush()
		}
		if err != nil {
			return
		}
	}
}

// applyProfileToOpenAI fills in alpakka's defaults for anything the caller did
// not set. An explicit value in the request always wins.
func applyProfileToOpenAI(body map[string]any, p config.Profile) {
	if p.ReasoningEffort != nil {
		kwargs, _ := body["chat_template_kwargs"].(map[string]any)
		if kwargs == nil {
			kwargs = map[string]any{}
		}
		if _, set := kwargs["reasoning_effort"]; !set {
			kwargs["reasoning_effort"] = *p.ReasoningEffort
		}
		body["chat_template_kwargs"] = kwargs
	}
	setIfAbsent(body, "temperature", p.Temperature)
	setIfAbsent(body, "top_p", p.TopP)
	setIfAbsent(body, "top_k", p.TopK)
	setIfAbsent(body, "min_p", p.MinP)
	setIfAbsent(body, "seed", p.Seed)

	// Ask for token counts on streams so clients that want usage get it.
	if stream, _ := body["stream"].(bool); stream {
		if _, ok := body["stream_options"]; !ok {
			body["stream_options"] = map[string]any{"include_usage": true}
		}
	}
}

func setIfAbsent[T any](body map[string]any, key string, v *T) {
	if v == nil {
		return
	}
	if _, ok := body[key]; ok {
		return
	}
	body[key] = *v
}

// optionsFrom lets an OpenAI-style caller reach alpakka's process-level
// settings through an "options" object, the same keys the ollama API uses.
func optionsFrom(body map[string]any) map[string]any {
	opts, _ := body["options"].(map[string]any)
	if opts != nil {
		delete(body, "options")
	}
	return opts
}

func keepAliveFrom(body map[string]any) *api.Duration {
	switch v := body["keep_alive"].(type) {
	case string:
		d, err := time.ParseDuration(v)
		if err != nil {
			return nil
		}
		return &api.Duration{Duration: d}
	case float64:
		return &api.Duration{Duration: time.Duration(v) * time.Second}
	}
	return nil
}

func writeOpenAIResolveError(w http.ResponseWriter, name string, err error) {
	var nf errModelNotFound
	var bad errBadRequest
	status, msg := http.StatusInternalServerError, err.Error()
	switch {
	case errorsAs(err, &nf):
		status, msg = http.StatusNotFound, "model '"+name+"' not found"
	case errorsAs(err, &bad):
		status = http.StatusBadRequest
	}
	// OpenAI clients read errors from a nested object.
	writeJSON(w, status, map[string]any{
		"error": map[string]any{"message": msg, "type": "invalid_request_error"},
	})
}

// handleOpenAIModels lists the store as OpenAI model objects.
func (s *Server) handleOpenAIModels(w http.ResponseWriter, r *http.Request) {
	models, err := s.Store.List()
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	data := make([]map[string]any, 0, len(models))
	for _, m := range models {
		data = append(data, map[string]any{
			"id":       m.Name,
			"object":   "model",
			"created":  m.ModifiedAt.Unix(),
			"owned_by": "alpakka",
		})
	}
	writeJSON(w, http.StatusOK, map[string]any{"object": "list", "data": data})
}
