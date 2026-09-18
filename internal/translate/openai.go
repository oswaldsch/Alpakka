// Package translate converts between ollama's wire format and the
// OpenAI-compatible API that llama-server speaks.
//
// alpakka never renders a chat template itself. llama-server runs with --jinja
// and applies the model's own template, because that is the only path on which
// chat_template_kwargs — and therefore reasoning_effort — exists. That leaves
// this package with a single job: shape translation.
package translate

import (
	"encoding/base64"
	"encoding/json"
	"time"

	"github.com/ollama/ollama/api"

	"github.com/oswald/alpakka/internal/config"
)

// Request is the subset of the OpenAI chat completion body alpakka sends.
type Request struct {
	Model    string    `json:"model"`
	Messages []Message `json:"messages"`
	Stream   bool      `json:"stream"`

	MaxTokens     *int      `json:"max_tokens,omitempty"`
	Temperature   *float32  `json:"temperature,omitempty"`
	TopP          *float32  `json:"top_p,omitempty"`
	TopK          *int      `json:"top_k,omitempty"`
	MinP          *float32  `json:"min_p,omitempty"`
	RepeatPenalty *float32  `json:"repeat_penalty,omitempty"`
	Seed          *int      `json:"seed,omitempty"`
	Stop          []string  `json:"stop,omitempty"`
	Tools         api.Tools `json:"tools,omitempty"`

	// llama-server accepts the rest of ollama's samplers under these exact
	// names, so they are forwarded rather than dropped.
	Mirostat         *int     `json:"mirostat,omitempty"`
	MirostatTau      *float32 `json:"mirostat_tau,omitempty"`
	MirostatEta      *float32 `json:"mirostat_eta,omitempty"`
	PresencePenalty  *float32 `json:"presence_penalty,omitempty"`
	FrequencyPenalty *float32 `json:"frequency_penalty,omitempty"`
	RepeatLastN      *int     `json:"repeat_last_n,omitempty"`
	TypicalP         *float32 `json:"typical_p,omitempty"`
	NumKeep          *int     `json:"n_keep,omitempty"`

	// CachePrompt and IgnoreEOS are llama.cpp's own completion parameters, set
	// only by the benchmark endpoint: reusing a cached prefix would report a
	// second run's prefill as free, and a model that stops early would measure
	// decode over a token count the caller did not choose.
	CachePrompt *bool `json:"cache_prompt,omitempty"`
	IgnoreEOS   *bool `json:"ignore_eos,omitempty"`

	ResponseFormat json.RawMessage `json:"response_format,omitempty"`

	// ChatTemplateKwargs is llama.cpp's hook into the jinja template. This is
	// where reasoning_effort goes, and it is the reason alpakka exists.
	ChatTemplateKwargs map[string]any `json:"chat_template_kwargs,omitempty"`

	StreamOptions *StreamOptions `json:"stream_options,omitempty"`
}

// StreamOptions asks llama-server for token counts in the final chunk.
type StreamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

// Message is an OpenAI chat message.
type Message struct {
	Role       string         `json:"role"`
	Content    any            `json:"content"`
	ToolCalls  []ToolCall     `json:"tool_calls,omitempty"`
	ToolCallID string         `json:"tool_call_id,omitempty"`
	Name       string         `json:"name,omitempty"`
	Extra      map[string]any `json:"-"`
}

// ToolCall is an OpenAI tool call.
type ToolCall struct {
	ID       string `json:"id,omitempty"`
	Type     string `json:"type,omitempty"`
	Index    *int   `json:"index,omitempty"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

// Chunk is one server-sent event from a streaming completion.
type Chunk struct {
	Choices []struct {
		Index        int    `json:"index"`
		FinishReason string `json:"finish_reason"`
		Delta        struct {
			Role string `json:"role"`
			// Content is null on the opening chunk, so it is a pointer.
			Content *string `json:"content"`
			// llama.cpp emits reasoning_content; ollama's own OpenAI surface
			// calls the same thing reasoning. Both are accepted.
			ReasoningContent *string    `json:"reasoning_content"`
			Reasoning        *string    `json:"reasoning"`
			ToolCalls        []ToolCall `json:"tool_calls"`
		} `json:"delta"`
		Message *struct {
			Role             string     `json:"role"`
			Content          string     `json:"content"`
			ReasoningContent string     `json:"reasoning_content"`
			Reasoning        string     `json:"reasoning"`
			ToolCalls        []ToolCall `json:"tool_calls"`
		} `json:"message"`
	} `json:"choices"`
	Usage   *Usage   `json:"usage"`
	Timings *Timings `json:"timings"`
}

// Usage is the OpenAI token accounting.
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
}

// Timings is llama.cpp's timing extension, which carries the durations ollama
// reports as prompt_eval_duration and eval_duration.
type Timings struct {
	PromptN     int     `json:"prompt_n"`
	PromptMS    float64 `json:"prompt_ms"`
	PredictedN  int     `json:"predicted_n"`
	PredictedMS float64 `json:"predicted_ms"`
}

// Metrics converts llama.cpp's counters into ollama's. totalDuration is the
// time spent generating; the load is added because ollama's total_duration
// covers the whole request.
func (c *Chunk) Metrics(loadDuration, totalDuration time.Duration) api.Metrics {
	m := api.Metrics{LoadDuration: loadDuration, TotalDuration: loadDuration + totalDuration}
	if c.Usage != nil {
		m.PromptEvalCount = c.Usage.PromptTokens
		m.EvalCount = c.Usage.CompletionTokens
	}
	if c.Timings != nil {
		if c.Timings.PromptN > 0 {
			m.PromptEvalCount = c.Timings.PromptN
		}
		if c.Timings.PredictedN > 0 {
			m.EvalCount = c.Timings.PredictedN
		}
		m.PromptEvalDuration = millis(c.Timings.PromptMS)
		m.EvalDuration = millis(c.Timings.PredictedMS)
	}
	return m
}

func millis(ms float64) time.Duration {
	return time.Duration(ms * float64(time.Millisecond))
}

// Text returns the content delta of a chunk, from either the streaming delta or
// a non-streaming message.
func (c *Chunk) Text() (content, thinking string) {
	if len(c.Choices) == 0 {
		return "", ""
	}
	ch := c.Choices[0]
	if ch.Message != nil {
		thinking = ch.Message.ReasoningContent
		if thinking == "" {
			thinking = ch.Message.Reasoning
		}
		return ch.Message.Content, thinking
	}
	if ch.Delta.Content != nil {
		content = *ch.Delta.Content
	}
	switch {
	case ch.Delta.ReasoningContent != nil:
		thinking = *ch.Delta.ReasoningContent
	case ch.Delta.Reasoning != nil:
		thinking = *ch.Delta.Reasoning
	}
	return content, thinking
}

// DoneReason maps an OpenAI finish_reason onto ollama's done_reason.
func DoneReason(finish string) string {
	switch finish {
	case "", "stop":
		return "stop"
	case "length":
		return "length"
	case "tool_calls", "function_call":
		return "stop"
	default:
		return finish
	}
}

// BuildRequest turns an ollama-shaped call into an OpenAI one.
func BuildRequest(model string, msgs []Message, p config.Profile, stream bool) *Request {
	r := &Request{
		Model:         model,
		Messages:      msgs,
		Stream:        stream,
		Temperature:   p.Temperature,
		TopP:          p.TopP,
		TopK:          p.TopK,
		MinP:          p.MinP,
		RepeatPenalty: p.RepeatPenalty,
		Seed:          p.Seed,
		Stop:          p.Stop,

		Mirostat:         p.Mirostat,
		MirostatTau:      p.MirostatTau,
		MirostatEta:      p.MirostatEta,
		PresencePenalty:  p.PresencePenalty,
		FrequencyPenalty: p.FrequencyPenalty,
		RepeatLastN:      p.RepeatLastN,
		TypicalP:         p.TypicalP,
		NumKeep:          p.NumKeep,
	}
	// Ollama uses -1 for "no limit"; OpenAI wants the field absent.
	if p.NumPredict != nil && *p.NumPredict >= 0 {
		r.MaxTokens = p.NumPredict
	}
	if p.ReasoningEffort != nil {
		r.ChatTemplateKwargs = map[string]any{"reasoning_effort": *p.ReasoningEffort}
	}
	if stream {
		// Without this the final chunk carries no token counts, and ollama's
		// clients would see zeroed eval_count fields.
		r.StreamOptions = &StreamOptions{IncludeUsage: true}
	}
	return r
}

// FromOllamaMessages converts ollama messages to OpenAI ones. Message.Thinking
// is never forwarded: OpenAI's message shape has no slot for a historical
// reasoning block, and the model's own template drops it there too.
func FromOllamaMessages(in []api.Message) []Message {
	out := make([]Message, 0, len(in))
	for _, m := range in {
		msg := Message{Role: m.Role, Content: m.Content, ToolCallID: m.ToolCallID, Name: m.ToolName}
		if len(m.Images) > 0 {
			msg.Content = contentWithImages(m.Content, m.Images)
		}
		for _, tc := range m.ToolCalls {
			var oc ToolCall
			oc.Type = "function"
			oc.Function.Name = tc.Function.Name
			if b, err := json.Marshal(tc.Function.Arguments); err == nil {
				oc.Function.Arguments = string(b)
			}
			msg.ToolCalls = append(msg.ToolCalls, oc)
		}
		out = append(out, msg)
	}
	return out
}

// contentWithImages renders an ollama message's images as OpenAI content parts.
func contentWithImages(text string, images []api.ImageData) []map[string]any {
	parts := []map[string]any{}
	if text != "" {
		parts = append(parts, map[string]any{"type": "text", "text": text})
	}
	for _, img := range images {
		parts = append(parts, map[string]any{
			"type":      "image_url",
			"image_url": map[string]any{"url": dataURL(img)},
		})
	}
	return parts
}

// dataURL base64-encodes image bytes for an OpenAI image_url part.
func dataURL(img api.ImageData) string {
	return "data:image/png;base64," + base64.StdEncoding.EncodeToString(img)
}
