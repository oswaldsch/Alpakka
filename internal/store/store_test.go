package store

import (
	"encoding/json"
	"os"
	"reflect"
	"testing"

	"github.com/ollama/ollama/api"
)

// testStore opens the real ollama store, skipping when it is not present.
func testStore(t *testing.T) *Store {
	t.Helper()
	root := DefaultRoot()
	if _, err := os.Stat(root); err != nil {
		t.Skip("ollama model store not present")
	}
	return New(root)
}

// golden is the /api/tags response captured from the real ollama on :11434.
func golden(t *testing.T) map[string]api.ListModelResponse {
	t.Helper()
	b, err := os.ReadFile("testdata/tags.json")
	if err != nil {
		t.Skip("no captured tags.json")
	}
	var resp api.ListResponse
	if err := json.Unmarshal(b, &resp); err != nil {
		t.Fatal(err)
	}
	out := map[string]api.ListModelResponse{}
	for _, m := range resp.Models {
		out[m.Name] = m
	}
	return out
}

// TestListMatchesOllama is the wire-fidelity test for /api/tags: every field
// alpakka derives from the store must equal what ollama itself reports.
func TestListMatchesOllama(t *testing.T) {
	s := testStore(t)
	want := golden(t)

	got, err := s.List()
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(want) {
		t.Errorf("listed %d models, ollama reports %d", len(got), len(want))
	}

	for _, m := range got {
		w, ok := want[m.Name]
		if !ok {
			t.Errorf("%s: not in ollama's listing", m.Name)
			continue
		}
		g := m.ListResponse()
		if g.Digest != w.Digest {
			t.Errorf("%s: digest = %s, want %s", m.Name, g.Digest, w.Digest)
		}
		if g.Size != w.Size {
			t.Errorf("%s: size = %d, want %d", m.Name, g.Size, w.Size)
		}
		if !g.ModifiedAt.Equal(w.ModifiedAt) {
			t.Errorf("%s: modified_at = %s, want %s", m.Name, g.ModifiedAt, w.ModifiedAt)
		}
		// Compared field by field so the one deliberate divergence below
		// stays visible rather than being buried in a struct diff.
		if g.Details.Format != w.Details.Format ||
			g.Details.Family != w.Details.Family ||
			!reflect.DeepEqual(g.Details.Families, w.Details.Families) ||
			g.Details.ParameterSize != w.Details.ParameterSize ||
			g.Details.QuantizationLevel != w.Details.QuantizationLevel ||
			g.Details.ParentModel != w.Details.ParentModel {
			t.Errorf("%s: details = %+v, want %+v", m.Name, g.Details, w.Details)
		}

		// context_length and embedding_length are read straight from the GGUF
		// header. Ollama reports 0 for the gemma4 omni models, whose headers
		// carry vision.* and audio.* sub-configs its reader does not handle.
		// Both keys are present and correct, so alpakka reports them; these
		// fields are omitempty extras that no client behaviour depends on, and
		// reproducing ollama's gap would be worse than diverging from it.
		if w.Details.ContextLength != 0 && g.Details.ContextLength != w.Details.ContextLength {
			t.Errorf("%s: context_length = %d, want %d",
				m.Name, g.Details.ContextLength, w.Details.ContextLength)
		}
		if w.Details.EmbeddingLength != 0 && g.Details.EmbeddingLength != w.Details.EmbeddingLength {
			t.Errorf("%s: embedding_length = %d, want %d",
				m.Name, g.Details.EmbeddingLength, w.Details.EmbeddingLength)
		}
		if w.Details.ContextLength == 0 && g.Details.ContextLength == 0 {
			t.Errorf("%s: ollama reports no context_length and neither do we; "+
				"the GGUF header should have supplied one", m.Name)
		}
	}
}

// TestListSortedNewestFirst matches ollama's ordering.
func TestListSortedNewestFirst(t *testing.T) {
	s := testStore(t)
	got, err := s.List()
	if err != nil {
		t.Fatal(err)
	}
	for i := 1; i < len(got); i++ {
		if got[i-1].ModifiedAt.Before(got[i].ModifiedAt) {
			t.Fatalf("model %d (%s) is older than %d (%s)",
				i-1, got[i-1].Name, i, got[i].Name)
		}
	}
}

func TestGetDefaultsTagToLatest(t *testing.T) {
	s := testStore(t)
	m, err := s.Get("qwen3.8-27b-q3-32k")
	if err != nil {
		t.Skipf("model not present: %v", err)
	}
	if m.Name != "qwen3.8-27b-q3-32k:latest" {
		t.Errorf("Name = %q", m.Name)
	}
	if m.ModelPath == "" {
		t.Error("no model path")
	}
	if m.ProjectorPath == "" {
		t.Error("expected a projector layer")
	}
	if got := m.Params["num_ctx"]; got != float64(32768) {
		t.Errorf("params num_ctx = %v (%T), want 32768", got, got)
	}
}

func TestGetUnknownModel(t *testing.T) {
	s := testStore(t)
	if _, err := s.Get("definitely-not-a-model"); err == nil {
		t.Fatal("expected an error")
	}
}

func TestCanonicalName(t *testing.T) {
	for _, c := range []struct{ rel, want string }{
		{"registry.ollama.ai/library/qwen3/0.6b", "qwen3:0.6b"},
		{"registry.ollama.ai/huihui_ai/qwen3.5-abliterated/9b", "huihui_ai/qwen3.5-abliterated:9b"},
		{"hf.co/unsloth/Qwen3.8-27B-GGUF/Q3_K_M", "hf.co/unsloth/Qwen3.8-27B-GGUF:Q3_K_M"},
	} {
		got, ok := canonicalName(c.rel)
		if !ok || got != c.want {
			t.Errorf("canonicalName(%q) = %q, %v; want %q", c.rel, got, ok, c.want)
		}
	}
}
