package api

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"strings"
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

	// RawMessage keeps untouched keys (messages, tools, embedding input, ...)
	// as unparsed bytes: a multimodal or embedding body never gets recursively
	// decoded and re-encoded just to patch five scalar fields.
	var body map[string]json.RawMessage
	if err := json.Unmarshal(raw, &body); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body: "+err.Error())
		return
	}
	name := rawString(body["model"])
	if name == "" {
		writeError(w, http.StatusBadRequest, "model is required")
		return
	}

	inst, profile, _, err := s.resolve(r.Context(), name, optionsFrom(body), keepAliveFrom(body),
		r.URL.Path == "/v1/embeddings")
	if err != nil {
		writeOpenAIResolveError(w, name, err)
		return
	}
	defer inst.Release()

	// llama-server knows the model by the alias it was started with.
	body["model"] = toRaw(inst.Runtime().Model)
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
		// alpakka's own CORS headers are already set; llama-server's would be
		// duplicated, and a duplicated Allow-Origin is rejected by browsers.
		if strings.HasPrefix(strings.ToLower(k), "access-control-") {
			continue
		}
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
func applyProfileToOpenAI(body map[string]json.RawMessage, p config.Profile) {
	if p.ReasoningEffort != nil {
		var kwargs map[string]any
		_ = json.Unmarshal(body["chat_template_kwargs"], &kwargs)
		if kwargs == nil {
			kwargs = map[string]any{}
		}
		if _, set := kwargs["reasoning_effort"]; !set {
			kwargs["reasoning_effort"] = *p.ReasoningEffort
		}
		body["chat_template_kwargs"] = toRaw(kwargs)
	}
	setIfAbsent(body, "temperature", p.Temperature)
	setIfAbsent(body, "top_p", p.TopP)
	setIfAbsent(body, "top_k", p.TopK)
	setIfAbsent(body, "min_p", p.MinP)
	setIfAbsent(body, "seed", p.Seed)

	// Ask for token counts on streams so clients that want usage get it.
	if rawBool(body["stream"]) {
		if _, ok := body["stream_options"]; !ok {
			body["stream_options"] = toRaw(map[string]any{"include_usage": true})
		}
	}
}

func setIfAbsent[T any](body map[string]json.RawMessage, key string, v *T) {
	if v == nil {
		return
	}
	if _, ok := body[key]; ok {
		return
	}
	body[key] = toRaw(*v)
}

// optionsFrom lets an OpenAI-style caller reach alpakka's process-level
// settings through an "options" object, the same keys the ollama API uses.
func optionsFrom(body map[string]json.RawMessage) map[string]any {
	raw, ok := body["options"]
	if !ok {
		return nil
	}
	var opts map[string]any
	if err := json.Unmarshal(raw, &opts); err != nil || opts == nil {
		return nil
	}
	delete(body, "options")
	return opts
}

func keepAliveFrom(body map[string]json.RawMessage) *api.Duration {
	var v any
	if err := json.Unmarshal(body["keep_alive"], &v); err != nil {
		return nil
	}
	switch v := v.(type) {
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

// rawString unmarshals a possibly-absent RawMessage as a string, treating
// anything that isn't one as empty rather than an error.
func rawString(v json.RawMessage) string {
	var s string
	_ = json.Unmarshal(v, &s)
	return s
}

// rawBool unmarshals a possibly-absent RawMessage as a bool.
func rawBool(v json.RawMessage) bool {
	var b bool
	_ = json.Unmarshal(v, &b)
	return b
}

// toRaw marshals a value alpakka built itself. These are always plain
// strings, numbers or maps of the same, which cannot fail to marshal.
func toRaw(v any) json.RawMessage {
	b, _ := json.Marshal(v)
	return b
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
