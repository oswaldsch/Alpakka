package translate

import (
	"encoding/json"
	"os"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

func fixture(t *testing.T, name string) *os.File {
	t.Helper()
	f, err := os.Open("testdata/" + name)
	if err != nil {
		t.Skipf("no fixture %s", name)
	}
	t.Cleanup(func() { f.Close() })
	return f
}

// testLoad stands in for the time Ensure spent getting a server ready. It is
// never zero in practice, and load_duration is omitempty, so passing zero here
// would drop a field ollama always sends.
const testLoad = 137 * time.Millisecond

// collectChat runs a captured llama.cpp stream through the translator.
func collectChat(t *testing.T, name string) []api.ChatResponse {
	t.Helper()
	var got []api.ChatResponse
	err := Chat(fixture(t, name), "qwen3:0.6b", testLoad, time.Now(), func(r api.ChatResponse) error {
		got = append(got, r)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	return got
}

// jsonKeys lists the field names an object actually serialises, which is what a
// client sees. Wire bugs live here.
func jsonKeys(t *testing.T, v any) []string {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]json.RawMessage
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatal(err)
	}
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// ollamaGolden returns the NDJSON captured from the real ollama server.
func ollamaGolden(t *testing.T) (first, last map[string]json.RawMessage) {
	t.Helper()
	b, err := os.ReadFile("testdata/chat_stream.ndjson")
	if err != nil {
		t.Skip("no captured ollama stream")
	}
	lines := strings.Split(strings.TrimSpace(string(b)), "\n")
	if err := json.Unmarshal([]byte(lines[0]), &first); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal([]byte(lines[len(lines)-1]), &last); err != nil {
		t.Fatal(err)
	}
	return first, last
}

// TestChatFramingMatchesOllama is the wire-compatibility test: the objects
// alpakka emits must carry the same field names, in the same places, as the
// ones a real ollama server emitted for the same kind of request.
func TestChatFramingMatchesOllama(t *testing.T) {
	got := collectChat(t, "llama_stream.sse")
	if len(got) < 2 {
		t.Fatalf("got %d responses, expected a stream plus a final", len(got))
	}
	first, last := got[0], got[len(got)-1]
	wantFirst, wantLast := ollamaGolden(t)

	keys := func(m map[string]json.RawMessage) []string {
		out := make([]string, 0, len(m))
		for k := range m {
			out = append(out, k)
		}
		sort.Strings(out)
		return out
	}

	if g, w := strings.Join(jsonKeys(t, first), ","), strings.Join(keys(wantFirst), ","); g != w {
		t.Errorf("streaming chunk fields:\n got %s\nwant %s", g, w)
	}
	if g, w := strings.Join(jsonKeys(t, last), ","), strings.Join(keys(wantLast), ","); g != w {
		t.Errorf("final chunk fields:\n got %s\nwant %s", g, w)
	}
}

func TestChatStreamShape(t *testing.T) {
	got := collectChat(t, "llama_stream.sse")

	for i, r := range got[:len(got)-1] {
		if r.Done {
			t.Errorf("chunk %d: done should be false", i)
		}
		if r.Message.Role != "assistant" {
			t.Errorf("chunk %d: role = %q", i, r.Message.Role)
		}
		if r.Message.Content == "" && r.Message.Thinking == "" {
			t.Errorf("chunk %d: emitted an empty delta; ollama emits none", i)
		}
		if r.CreatedAt.IsZero() {
			t.Errorf("chunk %d: no created_at", i)
		}
	}

	last := got[len(got)-1]
	if !last.Done {
		t.Error("final chunk: done = false")
	}
	if last.Message.Content != "" {
		t.Errorf("final chunk: content = %q, ollama sends an empty one", last.Message.Content)
	}
	if last.DoneReason == "" {
		t.Error("final chunk: no done_reason")
	}
	if last.EvalCount == 0 || last.PromptEvalCount == 0 {
		t.Errorf("final chunk: token counts are zero (%d/%d); stream_options.include_usage "+
			"is what makes llama.cpp report them", last.PromptEvalCount, last.EvalCount)
	}
	if last.EvalDuration == 0 || last.PromptEvalDuration == 0 {
		t.Error("final chunk: durations are zero; llama.cpp's timings block supplies them")
	}
}

// Reasoning arrives as reasoning_content from llama.cpp and has to surface as
// ollama's thinking field.
func TestThinkingIsTranslated(t *testing.T) {
	got := collectChat(t, "llama_stream_thinking.sse")
	var sawThinking bool
	for _, r := range got {
		if r.Message.Thinking != "" {
			sawThinking = true
		}
	}
	if !sawThinking {
		t.Error("no thinking content surfaced from reasoning_content deltas")
	}
}

func TestCollectChatJoinsTheStream(t *testing.T) {
	streamed := collectChat(t, "llama_stream.sse")
	var want strings.Builder
	for _, r := range streamed {
		want.WriteString(r.Message.Content)
	}

	got, err := CollectChat(fixture(t, "llama_stream.sse"), "qwen3:0.6b", testLoad, time.Now())
	if err != nil {
		t.Fatal(err)
	}
	if got.Message.Content != want.String() {
		t.Errorf("content = %q, want %q", got.Message.Content, want.String())
	}
	if !got.Done || got.DoneReason == "" {
		t.Error("non-streaming response must still be done with a reason")
	}
}

func TestGenerateStreamShape(t *testing.T) {
	var got []api.GenerateResponse
	err := Generate(fixture(t, "llama_stream.sse"), "qwen3:0.6b", testLoad, time.Now(),
		func(r api.GenerateResponse) error {
			got = append(got, r)
			return nil
		})
	if err != nil {
		t.Fatal(err)
	}
	last := got[len(got)-1]
	if !last.Done || last.Response != "" || last.DoneReason == "" {
		t.Errorf("final generate chunk = %+v", last)
	}
	if got[0].Response == "" {
		t.Error("first chunk carries no response text")
	}
}

func TestDoneReasonMapping(t *testing.T) {
	for in, want := range map[string]string{
		"stop": "stop", "length": "length", "tool_calls": "stop", "": "stop",
	} {
		if got := DoneReason(in); got != want {
			t.Errorf("DoneReason(%q) = %q, want %q", in, got, want)
		}
	}
}

// Tool call arguments arrive as JSON fragments across many chunks and only
// parse once concatenated.
func TestToolCallsAreReassembled(t *testing.T) {
	sse := `data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"get_weather","arguments":"{\"city\""}}]}}]}
data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"name":"","arguments":":\"Berlin\"}"}}]}}]}
data: {"choices":[{"index":0,"finish_reason":"tool_calls","delta":{}}],"usage":{"prompt_tokens":5,"completion_tokens":7}}
data: [DONE]
`
	var got []api.ChatResponse
	err := Chat(strings.NewReader(sse), "m", 0, time.Now(), func(r api.ChatResponse) error {
		got = append(got, r)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) < 2 {
		t.Fatalf("got %d chunks, want at least 2", len(got))
	}

	// Ollama's framing: the tool calls ride on their own chunk, and the final
	// one carries only done, done_reason and the metrics.
	last := got[len(got)-1]
	if !last.Done || last.DoneReason != "stop" {
		t.Errorf("final chunk: done=%v done_reason=%q", last.Done, last.DoneReason)
	}
	if len(last.Message.ToolCalls) != 0 {
		t.Errorf("final chunk carried %d tool calls; ollama leaves it empty",
			len(last.Message.ToolCalls))
	}

	calls := got[len(got)-2]
	if calls.Done {
		t.Error("the tool call chunk should not be the done chunk")
	}
	if len(calls.Message.ToolCalls) != 1 {
		t.Fatalf("got %d tool calls, want 1", len(calls.Message.ToolCalls))
	}
	call := calls.Message.ToolCalls[0]
	if call.Function.Name != "get_weather" {
		t.Errorf("name = %q", call.Function.Name)
	}
	if got, ok := call.Function.Arguments.Get("city"); !ok || got != "Berlin" {
		t.Errorf("arguments[city] = %v, want Berlin", got)
	}

	// The non-streaming form has to find them on that chunk, not the last one.
	resp, err := CollectChat(strings.NewReader(sse), "m", 0, time.Now())
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Message.ToolCalls) != 1 {
		t.Errorf("CollectChat lost the tool calls: %+v", resp.Message)
	}
}
