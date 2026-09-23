package api

import (
	"encoding/json"
	"net/http"
	"strings"
	"testing"
)

func TestBenchRejectsBadRequests(t *testing.T) {
	h := testServer(t)
	cases := []struct{ name, body, want string }{
		{"no model", `{}`, "model is required"},
		{"both prompts", `{"model":"qwen3:0.6b","prompt":"hi","prompt_tokens":8}`, "not both"},
		{"too many runs", `{"model":"qwen3:0.6b","runs":999}`, "runs must be"},
		{"prompt too long", `{"model":"qwen3:0.6b","prompt_tokens":99999999}`, "prompt_tokens must be"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			w := do(t, h, http.MethodPost, "/alpakka/bench", c.body)
			if w.Code != http.StatusBadRequest {
				t.Fatalf("status = %d, want 400 (body: %s)", w.Code, w.Body.String())
			}
			if !strings.Contains(w.Body.String(), c.want) {
				t.Errorf("error = %s, want it to mention %q", w.Body.String(), c.want)
			}
		})
	}
}

func TestBenchRejectsBadOptionsBeforeLoading(t *testing.T) {
	w := do(t, testServer(t), http.MethodPost, "/alpakka/bench",
		`{"model":"qwen3:0.6b","options":{"reasoning_effort":"high"}}`)
	if w.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400 (body: %s)", w.Code, w.Body.String())
	}
	if !strings.Contains(w.Body.String(), "xhigh") {
		t.Errorf("error should name the accepted values, got %s", w.Body.String())
	}
}

func TestBenchUnknownModel(t *testing.T) {
	w := do(t, testServer(t), http.MethodPost, "/alpakka/bench", `{"model":"definitely-absent"}`)
	if w.Code != http.StatusNotFound {
		t.Errorf("status = %d, want 404", w.Code)
	}
}

func TestBenchIsNotOnTheOllamaOrOpenAISurface(t *testing.T) {
	h := testServer(t)
	for _, p := range []string{"/api/bench", "/v1/bench", "/api/alpakka/bench"} {
		if w := do(t, h, http.MethodPost, p, `{"model":"qwen3:0.6b"}`); w.Code != http.StatusNotFound {
			t.Errorf("%s: status = %d, want 404", p, w.Code)
		}
	}
}

func TestBenchStatusReportsNothingLoaded(t *testing.T) {
	w := do(t, testServer(t), http.MethodGet, "/alpakka/status", "")
	var got map[string]any
	if err := json.Unmarshal(w.Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got["loaded"] != false {
		t.Errorf("status = %s", w.Body.String())
	}
}

// The word count only estimates tokens, but a prompt asked for in the thousands must not come back in the tens.
func TestFillerPromptLengthAndVariation(t *testing.T) {
	a := fillerPrompt(2000, 0)
	if got := len(strings.Fields(a)); got != 2000 {
		t.Errorf("words = %d, want 2000", got)
	}
	// Each run gets its own prompt so llama.cpp cannot answer the second prefill from the first run's cache.
	if b := fillerPrompt(2000, 1); a == b {
		t.Error("two runs were given the same prompt")
	}
}

func TestAggregatePoolsRunsRatherThanAveragingRates(t *testing.T) {
	got := aggregate([]benchRun{
		{PromptTokens: 100, PromptMS: 100, PredictTokens: 10, PredictMS: 1000},
		{PromptTokens: 900, PromptMS: 900, PredictTokens: 90, PredictMS: 1000},
	})
	if got.PromptTokens != 1000 || got.PromptTPS != 1000 {
		t.Errorf("prompt = %d tokens at %v tok/s, want 1000 at 1000", got.PromptTokens, got.PromptTPS)
	}
	// Averaging the two rates would give 32.5, pooling gives the real 50.
	if got.PredictTPS != 50 {
		t.Errorf("predict_tps = %v, want 50", got.PredictTPS)
	}
}
