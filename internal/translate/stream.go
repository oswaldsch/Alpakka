package translate

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
)

// sseData is the prefix of a server-sent event's payload line.
const sseData = "data: "

// ReadSSE parses a server-sent event stream, calling fn for each chunk. The
// terminating "[DONE]" event ends the stream without invoking fn.
func ReadSSE(r io.Reader, fn func(*Chunk) error) error {
	sc := bufio.NewScanner(r)
	// Chunks stay small, but a tool call with large arguments can push a single
	// event well past the default limit.
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)

	for sc.Scan() {
		line := sc.Text()
		if !strings.HasPrefix(line, sseData) {
			continue
		}
		payload := strings.TrimSpace(line[len(sseData):])
		if payload == "" {
			continue
		}
		if payload == "[DONE]" {
			return nil
		}

		var c Chunk
		if err := json.Unmarshal([]byte(payload), &c); err != nil {
			return fmt.Errorf("decoding stream chunk: %w", err)
		}
		if err := fn(&c); err != nil {
			return err
		}
	}
	return sc.Err()
}

// Accumulator collects the parts of a completion that only arrive at the end:
// the finish reason, the token counts, and any streamed tool call fragments.
type Accumulator struct {
	FinishReason string
	Metrics      api.Metrics
	tools        map[int]*toolBuilder
	order        []int
}

type toolBuilder struct {
	id   string
	name string
	args strings.Builder
}

// Observe folds one chunk into the accumulator.
func (a *Accumulator) Observe(c *Chunk, load, total time.Duration) {
	if len(c.Choices) > 0 && c.Choices[0].FinishReason != "" {
		a.FinishReason = c.Choices[0].FinishReason
	}
	if c.Usage != nil || c.Timings != nil {
		a.Metrics = c.Metrics(load, total)
	}
	a.observeTools(c)
}

func (a *Accumulator) observeTools(c *Chunk) {
	if len(c.Choices) == 0 {
		return
	}
	calls := c.Choices[0].Delta.ToolCalls
	if c.Choices[0].Message != nil {
		calls = c.Choices[0].Message.ToolCalls
	}
	for i, tc := range calls {
		idx := i
		if tc.Index != nil {
			idx = *tc.Index
		}
		if a.tools == nil {
			a.tools = map[int]*toolBuilder{}
		}
		b, ok := a.tools[idx]
		if !ok {
			b = &toolBuilder{}
			a.tools[idx] = b
			a.order = append(a.order, idx)
		}
		if tc.ID != "" {
			b.id = tc.ID
		}
		if tc.Function.Name != "" {
			b.name = tc.Function.Name
		}
		// Arguments arrive as a stream of JSON fragments that only parse once
		// concatenated, so they are joined and decoded at the end.
		b.args.WriteString(tc.Function.Arguments)
	}
}

// ToolCalls returns the assembled tool calls, in the order they first appeared.
func (a *Accumulator) ToolCalls() []api.ToolCall {
	if len(a.order) == 0 {
		return nil
	}
	out := make([]api.ToolCall, 0, len(a.order))
	for i, idx := range a.order {
		b := a.tools[idx]
		call := api.ToolCall{ID: b.id}
		call.Function.Index = i
		call.Function.Name = b.name

		raw := strings.TrimSpace(b.args.String())
		if raw == "" {
			raw = "{}"
		}
		if err := json.Unmarshal([]byte(raw), &call.Function.Arguments); err != nil {
			// A tool call whose arguments do not parse is not usable, and
			// inventing a value would be worse than dropping it.
			continue
		}
		out = append(out, call)
	}
	return out
}

// DoneReason is the accumulated finish reason in ollama's vocabulary.
func (a *Accumulator) DoneReason() string { return DoneReason(a.FinishReason) }

// Chat converts an OpenAI SSE stream into ollama /api/chat responses, calling
// emit for each one. The final response carries done, done_reason and metrics,
// matching ollama's framing exactly.
func Chat(r io.Reader, model string, load time.Duration, start time.Time, emit func(api.ChatResponse) error) error {
	var acc Accumulator

	err := ReadSSE(r, func(c *Chunk) error {
		acc.Observe(c, load, time.Since(start))
		content, thinking := c.Text()
		if content == "" && thinking == "" {
			// llama.cpp opens with a role-only delta and closes with a
			// usage-only event; ollama emits neither.
			return nil
		}
		return emit(api.ChatResponse{
			Model:     model,
			CreatedAt: time.Now().UTC(),
			Message:   api.Message{Role: "assistant", Content: content, Thinking: thinking},
			Done:      false,
		})
	})
	if err != nil {
		return err
	}

	acc.Metrics.TotalDuration = time.Since(start)
	acc.Metrics.LoadDuration = load
	return emit(api.ChatResponse{
		Model:      model,
		CreatedAt:  time.Now().UTC(),
		Message:    api.Message{Role: "assistant", ToolCalls: acc.ToolCalls()},
		Done:       true,
		DoneReason: acc.DoneReason(),
		Metrics:    acc.Metrics,
	})
}

// Generate converts an OpenAI SSE stream into ollama /api/generate responses.
func Generate(r io.Reader, model string, load time.Duration, start time.Time, emit func(api.GenerateResponse) error) error {
	var acc Accumulator

	err := ReadSSE(r, func(c *Chunk) error {
		acc.Observe(c, load, time.Since(start))
		content, thinking := c.Text()
		if content == "" && thinking == "" {
			return nil
		}
		return emit(api.GenerateResponse{
			Model:     model,
			CreatedAt: time.Now().UTC(),
			Response:  content,
			Thinking:  thinking,
			Done:      false,
		})
	})
	if err != nil {
		return err
	}

	acc.Metrics.TotalDuration = time.Since(start)
	acc.Metrics.LoadDuration = load
	return emit(api.GenerateResponse{
		Model:      model,
		CreatedAt:  time.Now().UTC(),
		Done:       true,
		DoneReason: acc.DoneReason(),
		ToolCalls:  acc.ToolCalls(),
		Metrics:    acc.Metrics,
	})
}

// CollectChat folds a whole stream into the single response shape ollama
// returns when stream is false.
func CollectChat(r io.Reader, model string, load time.Duration, start time.Time) (api.ChatResponse, error) {
	var content, thinking bytes.Buffer
	var final api.ChatResponse

	err := Chat(r, model, load, start, func(resp api.ChatResponse) error {
		if !resp.Done {
			content.WriteString(resp.Message.Content)
			thinking.WriteString(resp.Message.Thinking)
			return nil
		}
		final = resp
		return nil
	})
	if err != nil {
		return final, err
	}
	final.Message.Content = content.String()
	final.Message.Thinking = thinking.String()
	return final, nil
}

// CollectGenerate is the non-streaming form of Generate.
func CollectGenerate(r io.Reader, model string, load time.Duration, start time.Time) (api.GenerateResponse, error) {
	var content, thinking bytes.Buffer
	var final api.GenerateResponse

	err := Generate(r, model, load, start, func(resp api.GenerateResponse) error {
		if !resp.Done {
			content.WriteString(resp.Response)
			thinking.WriteString(resp.Thinking)
			return nil
		}
		final = resp
		return nil
	})
	if err != nil {
		return final, err
	}
	final.Response = content.String()
	final.Thinking = thinking.String()
	return final, nil
}
