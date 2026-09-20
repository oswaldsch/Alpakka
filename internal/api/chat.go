package api

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"sort"
	"time"

	"github.com/ollama/ollama/api"

	"github.com/oswald/alpakka/internal/gguf"
	"github.com/oswald/alpakka/internal/store"
	"github.com/oswald/alpakka/internal/translate"
)

// upstream carries generation requests to llama-server. Generation has no
// meaningful timeout: a long completion at 25 tokens a second is normal.
var upstream = &http.Client{}

func (s *Server) handleChat(w http.ResponseWriter, r *http.Request) {
	var req api.ChatRequest
	if !decode(w, r, &req) {
		return
	}

	inst, profile, load, err := s.resolve(r.Context(), req.Model, req.Options, req.KeepAlive, false)
	if err != nil {
		writeResolveError(w, req.Model, err)
		return
	}
	// The instance is pinned until this returns, so neither the evictor nor a
	// request for another model can kill the process mid-stream.
	defer inst.Release()

	stream := req.Stream == nil || *req.Stream
	// llama-server is always asked to stream, even when the client wants a
	// single response. Collecting a stream is one well-tested code path;
	// supporting both of llama-server's response shapes would be two.
	body := translate.BuildRequest(inst.Runtime().Model,
		translate.FromOllamaMessages(req.Messages), profile, true)
	body.Tools = req.Tools
	applyThink(body, req.Think)
	applyFormat(body, req.Format)

	resp, err := s.post(r, inst.BaseURL()+"/v1/chat/completions", body)
	if err != nil {
		writeError(w, http.StatusBadGateway, err.Error())
		return
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		relayUpstreamError(w, resp)
		return
	}

	start := time.Now()
	model := inst.Runtime().Model

	if !stream {
		out, err := translate.CollectChat(resp.Body, model, load, start)
		if err != nil {
			writeError(w, http.StatusInternalServerError, err.Error())
			return
		}
		writeJSON(w, http.StatusOK, out)
		return
	}

	send := startNDJSON(w)
	err = translate.Chat(resp.Body, model, load, start, func(chunk api.ChatResponse) error {
		return send(chunk)
	})
	if err != nil {
		s.logf("chat stream ended early: %v", err)
	}
}

func (s *Server) handleGenerate(w http.ResponseWriter, r *http.Request) {
	var req api.GenerateRequest
	if !decode(w, r, &req) {
		return
	}

	inst, profile, load, err := s.resolve(r.Context(), req.Model, req.Options, req.KeepAlive, false)
	if err != nil {
		writeResolveError(w, req.Model, err)
		return
	}
	// The instance is pinned until this returns, so neither the evictor nor a
	// request for another model can kill the process mid-stream.
	defer inst.Release()

	// /api/generate is a single-turn chat. Sending it through the chat endpoint
	// keeps the model's own template — and reasoning_effort — in play, which a
	// raw completion would bypass.
	msgs := []translate.Message{}
	if req.System != "" {
		msgs = append(msgs, translate.Message{Role: "system", Content: req.System})
	}
	user := translate.Message{Role: "user", Content: req.Prompt}
	if len(req.Images) > 0 {
		user.Content = imageParts(req.Prompt, req.Images)
	}
	msgs = append(msgs, user)

	stream := req.Stream == nil || *req.Stream
	body := translate.BuildRequest(inst.Runtime().Model, msgs, profile, true)
	applyThink(body, req.Think)
	applyFormat(body, req.Format)

	resp, err := s.post(r, inst.BaseURL()+"/v1/chat/completions", body)
	if err != nil {
		writeError(w, http.StatusBadGateway, err.Error())
		return
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		relayUpstreamError(w, resp)
		return
	}

	start := time.Now()
	model := inst.Runtime().Model

	if !stream {
		out, err := translate.CollectGenerate(resp.Body, model, load, start)
		if err != nil {
			writeError(w, http.StatusInternalServerError, err.Error())
			return
		}
		writeJSON(w, http.StatusOK, out)
		return
	}

	send := startNDJSON(w)
	err = translate.Generate(resp.Body, model, load, start, func(chunk api.GenerateResponse) error {
		return send(chunk)
	})
	if err != nil {
		s.logf("generate stream ended early: %v", err)
	}
}

// startNDJSON prepares an ollama streaming response and returns a writer that
// emits one JSON object per line, flushed immediately so clients see tokens as
// they are produced.
func startNDJSON(w http.ResponseWriter) func(any) error {
	w.Header().Set("Content-Type", "application/x-ndjson")
	w.Header().Set("Cache-Control", "no-store")
	w.WriteHeader(http.StatusOK)

	enc := json.NewEncoder(w)
	flusher, _ := w.(http.Flusher)

	return func(v any) error {
		if err := enc.Encode(v); err != nil {
			return err
		}
		if flusher != nil {
			flusher.Flush()
		}
		return nil
	}
}

// applyThink maps ollama's think field onto the jinja template's kwargs.
func applyThink(body *translate.Request, think *api.ThinkValue) {
	if think == nil || !think.IsValid() {
		return
	}
	if body.ChatTemplateKwargs == nil {
		body.ChatTemplateKwargs = map[string]any{}
	}
	if think.IsBool() {
		body.ChatTemplateKwargs["enable_thinking"] = think.Bool()
		return
	}
	// A string think value is a reasoning effort. "high" is not a value the
	// Qwen3.8 template accepts; it silently becomes xhigh, so alpakka does that
	// mapping openly rather than letting the template do it quietly.
	effort := think.String()
	if effort == "high" {
		effort = "xhigh"
	}
	translate.SetReasoningEffort(body.ChatTemplateKwargs, effort)
}

// applyFormat maps ollama's format field onto an OpenAI response_format.
func applyFormat(body *translate.Request, format json.RawMessage) {
	if len(format) == 0 {
		return
	}
	var asString string
	if err := json.Unmarshal(format, &asString); err == nil {
		if asString == "json" {
			body.ResponseFormat = json.RawMessage(`{"type":"json_object"}`)
		}
		return
	}
	// Anything else is a JSON schema.
	body.ResponseFormat = json.RawMessage(
		`{"type":"json_schema","json_schema":{"name":"response","strict":true,"schema":` +
			string(format) + `}}`)
}

func imageParts(text string, images []api.ImageData) []map[string]any {
	msgs := translate.FromOllamaMessages([]api.Message{{Role: "user", Content: text, Images: images}})
	if parts, ok := msgs[0].Content.([]map[string]any); ok {
		return parts
	}
	return nil
}

// post sends a translated request upstream, propagating cancellation so a
// client hanging up stops the generation.
func (s *Server) post(r *http.Request, url string, body any) (*http.Response, error) {
	b, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(r.Context(), http.MethodPost, url, bytes.NewReader(b))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	return upstream.Do(req)
}

// relayUpstreamError forwards a llama-server error rather than inventing one.
func relayUpstreamError(w http.ResponseWriter, resp *http.Response) {
	e := upstreamError(resp)
	writeError(w, e.status, e.Error())
}

// errUpstream carries a llama-server failure back through a call chain that
// cannot write the response itself, keeping the status llama-server chose.
type errUpstream struct {
	status int
	msg    string
}

func (e errUpstream) Error() string { return e.msg }

func upstreamError(resp *http.Response) errUpstream {
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 64*1024))

	var wrapped struct {
		Error struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	msg := string(bytes.TrimSpace(body))
	if err := json.Unmarshal(body, &wrapped); err == nil && wrapped.Error.Message != "" {
		msg = wrapped.Error.Message
	}
	return errUpstream{status: resp.StatusCode, msg: "llama-server: " + msg}
}

func sortedKeys[V any](m map[string]V) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

func errorsAs(err error, target any) bool { return errors.As(err, target) }

// modelfile renders the Modelfile `ollama show --modelfile` would print, which
// is what the benchmark harness greps to resolve a model's blob path.
func modelfile(m *store.Model) string {
	var b bytes.Buffer
	fmt.Fprintf(&b, "# Modelfile generated by \"alpakka show\"\n")
	fmt.Fprintf(&b, "# To build a new Modelfile based on this, replace FROM with:\n")
	fmt.Fprintf(&b, "# FROM %s\n\n", m.Name)
	fmt.Fprintf(&b, "FROM %s\n", m.ModelPath)
	if m.Template != "" {
		fmt.Fprintf(&b, "TEMPLATE \"\"\"%s\"\"\"\n", m.Template)
	}
	if m.System != "" {
		fmt.Fprintf(&b, "SYSTEM \"\"\"%s\"\"\"\n", m.System)
	}
	for _, k := range sortedKeys(m.Params) {
		switch v := m.Params[k].(type) {
		case []any:
			for _, e := range v {
				fmt.Fprintf(&b, "PARAMETER %s %s\n", k, quoteParam(e))
			}
		default:
			fmt.Fprintf(&b, "PARAMETER %s %s\n", k, quoteParam(v))
		}
	}
	return b.String()
}

// modelInfo renders the GGUF header as /api/show's model_info map.
func modelInfo(f *gguf.File) map[string]any {
	out := make(map[string]any, len(f.KV))
	for k, v := range f.KV {
		// The chat template is returned in its own field and is large.
		if k == "tokenizer.chat_template" {
			continue
		}
		out[k] = v
	}
	return out
}

func (s *Server) handleEmbeddings(w http.ResponseWriter, r *http.Request) {
	var req api.EmbeddingRequest
	if !decode(w, r, &req) {
		return
	}
	vectors, err := s.embed(r, req.Model, []string{req.Prompt}, req.Options, req.KeepAlive)
	if err != nil {
		writeResolveError(w, req.Model, err)
		return
	}
	// The legacy endpoint returns a single float64 vector.
	out := api.EmbeddingResponse{Embedding: make([]float64, 0)}
	if len(vectors) > 0 {
		out.Embedding = make([]float64, len(vectors[0]))
		for i, v := range vectors[0] {
			out.Embedding[i] = float64(v)
		}
	}
	writeJSON(w, http.StatusOK, out)
}

func (s *Server) handleEmbed(w http.ResponseWriter, r *http.Request) {
	var req api.EmbedRequest
	if !decode(w, r, &req) {
		return
	}
	var inputs []string
	switch v := req.Input.(type) {
	case string:
		inputs = []string{v}
	case []any:
		for _, e := range v {
			str, ok := e.(string)
			if !ok {
				writeError(w, http.StatusBadRequest, "input must be a string or list of strings")
				return
			}
			inputs = append(inputs, str)
		}
	default:
		writeError(w, http.StatusBadRequest, "input must be a string or list of strings")
		return
	}

	start := time.Now()
	vectors, err := s.embed(r, req.Model, inputs, req.Options, req.KeepAlive)
	if err != nil {
		writeResolveError(w, req.Model, err)
		return
	}
	writeJSON(w, http.StatusOK, api.EmbedResponse{
		Model:         req.Model,
		Embeddings:    vectors,
		TotalDuration: time.Since(start),
	})
}

func (s *Server) embed(r *http.Request, model string, inputs []string,
	opts map[string]any, keepAlive *api.Duration) ([][]float32, error) {

	inst, _, _, err := s.resolve(r.Context(), model, opts, keepAlive, true)
	if err != nil {
		return nil, err
	}
	defer inst.Release()

	resp, err := s.post(r, inst.BaseURL()+"/v1/embeddings", map[string]any{
		"model": inst.Runtime().Model,
		"input": inputs,
	})
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 8192))
		return nil, fmt.Errorf("llama-server: %s", bytes.TrimSpace(body))
	}

	var parsed struct {
		Data []struct {
			Embedding []float32 `json:"embedding"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&parsed); err != nil {
		return nil, err
	}
	out := make([][]float32, 0, len(parsed.Data))
	for _, d := range parsed.Data {
		out = append(out, d.Embedding)
	}
	return out, nil
}
