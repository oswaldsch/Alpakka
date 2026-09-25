package translate

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
)

const sseData = "data: "

// The terminating "[DONE]" event ends the stream without invoking fn.
// ErrTruncated is a stream that ended without saying it was done, as when llama-server is
// killed mid-response. Without it, a cut-off answer reads as a complete one.
var ErrTruncated = errors.New("llama-server's stream ended before the response was finished")

func ReadSSE(r io.Reader, fn func(*Chunk) error) error {
	sc := bufio.NewScanner(r)
	// A tool call with large arguments can push a single event past the default scanner limit.
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)

	finished := false
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
		for _, ch := range c.Choices {
			finished = finished || ch.FinishReason != ""
		}
		if err := fn(&c); err != nil {
			return err
		}
	}
	if err := sc.Err(); err != nil {
		return err
	}
	if !finished {
		return ErrTruncated
	}
	return nil
}

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
		// Arguments arrive as JSON fragments that only parse once concatenated.
		b.args.WriteString(tc.Function.Arguments)
	}
}

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
			// A tool call whose arguments do not parse is unusable, and inventing a value would be worse than dropping it.
			continue
		}
		out = append(out, call)
	}
	return out
}

func (a *Accumulator) DoneReason() string { return DoneReason(a.FinishReason) }

func Chat(r io.Reader, model string, load time.Duration, start time.Time, emit func(api.ChatResponse) error) error {
	var acc Accumulator

	err := ReadSSE(r, func(c *Chunk) error {
		acc.Observe(c, load, time.Since(start))
		content, thinking := c.Text()
		if content == "" && thinking == "" {
			// llama.cpp opens with a role-only delta and closes with a usage-only event, ollama emits neither.
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

	// Ollama delivers tool calls in their own chunk before the final one and leaves the final
	// message empty, so a client reading tool_calls from non-final chunks would miss them otherwise.
	if calls := acc.ToolCalls(); len(calls) > 0 {
		err := emit(api.ChatResponse{
			Model:     model,
			CreatedAt: time.Now().UTC(),
			Message:   api.Message{Role: "assistant", ToolCalls: calls},
			Done:      false,
		})
		if err != nil {
			return err
		}
	}

	// Ollama's total_duration spans the whole request, so a cold start must not report a total below its load_duration.
	acc.Metrics.TotalDuration = load + time.Since(start)
	acc.Metrics.LoadDuration = load
	return emit(api.ChatResponse{
		Model:      model,
		CreatedAt:  time.Now().UTC(),
		Message:    api.Message{Role: "assistant"},
		Done:       true,
		DoneReason: acc.DoneReason(),
		Metrics:    acc.Metrics,
	})
}

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

	// Ollama's total_duration spans the whole request, so a cold start must not report a total below its load_duration.
	acc.Metrics.TotalDuration = load + time.Since(start)
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

func CollectChat(r io.Reader, model string, load time.Duration, start time.Time) (api.ChatResponse, error) {
	var content, thinking bytes.Buffer
	var final api.ChatResponse

	var calls []api.ToolCall
	err := Chat(r, model, load, start, func(resp api.ChatResponse) error {
		if !resp.Done {
			content.WriteString(resp.Message.Content)
			thinking.WriteString(resp.Message.Thinking)
			// Tool calls arrive in their own non-final chunk, so the collected form picks them up there.
			calls = append(calls, resp.Message.ToolCalls...)
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
	final.Message.ToolCalls = calls
	return final, nil
}

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
