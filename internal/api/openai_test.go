package api

import (
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/oswald/alpakka/internal/config"
)

// Only model/options/keep_alive/profile are inspected, so a large unrelated field like messages
// must survive byte-for-byte.
func TestOpenAIBodyPreservesUntouchedFields(t *testing.T) {
	big := make([]map[string]any, 0, 500)
	for i := 0; i < 500; i++ {
		big = append(big, map[string]any{"role": "user", "content": strings.Repeat("x", 200)})
	}
	raw, err := json.Marshal(map[string]any{
		"model":      "qwen3.8-27b-q3-32k",
		"messages":   big,
		"stream":     true,
		"keep_alive": "5m",
		"options":    map[string]any{"num_ctx": 8192},
	})
	if err != nil {
		t.Fatal(err)
	}

	var body map[string]json.RawMessage
	if err := json.Unmarshal(raw, &body); err != nil {
		t.Fatal(err)
	}
	origMessages := string(body["messages"])

	opts := optionsFrom(body)
	if opts["num_ctx"] != float64(8192) {
		t.Fatalf("options: got %v", opts)
	}
	if _, ok := body["options"]; ok {
		t.Fatal("options key should have been removed")
	}

	ka := keepAliveFrom(body)
	if ka == nil || ka.Duration != 5*time.Minute {
		t.Fatalf("keep_alive: got %v", ka)
	}

	body["model"] = toRaw("resolved-alias")
	p := config.Profile{
		Temperature:     ptr(float32(0.7)),
		ReasoningEffort: ptr("low"),
	}
	applyProfileToOpenAI(body, p)
	delete(body, "keep_alive")

	if string(body["messages"]) != origMessages {
		t.Fatal("untouched messages field was mutated")
	}

	patched, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}

	var final map[string]any
	if err := json.Unmarshal(patched, &final); err != nil {
		t.Fatal(err)
	}
	if final["model"] != "resolved-alias" {
		t.Fatalf("model: got %v", final["model"])
	}
	if final["temperature"] != float64(0.7) {
		t.Fatalf("temperature: got %v", final["temperature"])
	}
	kwargs, _ := final["chat_template_kwargs"].(map[string]any)
	if kwargs["reasoning_effort"] != "low" {
		t.Fatalf("chat_template_kwargs: got %v", final["chat_template_kwargs"])
	}
	streamOpts, _ := final["stream_options"].(map[string]any)
	if streamOpts["include_usage"] != true {
		t.Fatalf("stream_options: got %v", final["stream_options"])
	}
	if _, ok := final["keep_alive"]; ok {
		t.Fatal("keep_alive leaked into the patched body")
	}
	msgs, _ := final["messages"].([]any)
	if len(msgs) != 500 {
		t.Fatalf("messages: got %d entries", len(msgs))
	}
}

func TestApplyProfileToOpenAIExplicitValueWins(t *testing.T) {
	body := map[string]json.RawMessage{
		"temperature":          toRaw(0.9),
		"chat_template_kwargs": toRaw(map[string]any{"reasoning_effort": "xhigh"}),
	}
	applyProfileToOpenAI(body, config.Profile{
		Temperature:     ptr(float32(0.1)),
		ReasoningEffort: ptr("low"),
	})

	var temp float64
	if err := json.Unmarshal(body["temperature"], &temp); err != nil || temp != 0.9 {
		t.Fatalf("temperature: got %v, err %v", temp, err)
	}
	var kwargs map[string]any
	if err := json.Unmarshal(body["chat_template_kwargs"], &kwargs); err != nil {
		t.Fatal(err)
	}
	if kwargs["reasoning_effort"] != "xhigh" {
		t.Fatalf("reasoning_effort: got %v", kwargs["reasoning_effort"])
	}
}

func TestOptionsFromAbsent(t *testing.T) {
	body := map[string]json.RawMessage{}
	if opts := optionsFrom(body); opts != nil {
		t.Fatalf("expected nil, got %v", opts)
	}
}

func TestKeepAliveFromVariants(t *testing.T) {
	cases := []struct {
		name string
		body map[string]json.RawMessage
		want *time.Duration
	}{
		{"absent", map[string]json.RawMessage{}, nil},
		{"string", map[string]json.RawMessage{"keep_alive": toRaw("30s")}, durPtr(30 * time.Second)},
		{"seconds", map[string]json.RawMessage{"keep_alive": toRaw(45)}, durPtr(45 * time.Second)},
		{"null", map[string]json.RawMessage{"keep_alive": toRaw(nil)}, nil},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := keepAliveFrom(c.body)
			switch {
			case c.want == nil && got != nil:
				t.Fatalf("expected nil, got %v", got)
			case c.want != nil && (got == nil || got.Duration != *c.want):
				t.Fatalf("expected %v, got %v", *c.want, got)
			}
		})
	}
}

func durPtr(d time.Duration) *time.Duration { return &d }
