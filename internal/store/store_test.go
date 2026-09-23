package store

import (
	"encoding/json"
	"os"
	"reflect"
	"sort"
	"testing"

	"github.com/ollama/ollama/api"
)

// An empty store is a skip, not a failure: the directory exists on any machine that ever ran ollama.
func testStore(t *testing.T) *Store {
	t.Helper()
	root := DefaultRoot()
	if _, err := os.Stat(root); err != nil {
		t.Skip("ollama model store not present")
	}
	s := New(root)
	models, err := s.List()
	if err != nil {
		t.Skipf("ollama model store not readable: %v", err)
	}
	if len(models) == 0 {
		t.Skip("ollama model store holds no models")
	}
	return s
}

// The /api/tags response captured from the real ollama on :11434.
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

// Only meaningful on the machine tags.json was captured from. Elsewhere the count and
// modified_at legitimately differ, so it skips.
func TestListMatchesOllama(t *testing.T) {
	s := testStore(t)
	want := golden(t)

	got, err := s.List()
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(want) {
		t.Skipf("tags.json describes a different store: it has %d models, this one has %d",
			len(want), len(got))
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
		// Compared field by field so the one deliberate divergence stays visible.
		if g.Details.Format != w.Details.Format ||
			g.Details.Family != w.Details.Family ||
			!reflect.DeepEqual(g.Details.Families, w.Details.Families) ||
			g.Details.ParameterSize != w.Details.ParameterSize ||
			g.Details.QuantizationLevel != w.Details.QuantizationLevel ||
			g.Details.ParentModel != w.Details.ParentModel {
			t.Errorf("%s: details = %+v, want %+v", m.Name, g.Details, w.Details)
		}

		// Ollama reports 0 for these on gemma4 omni models, whose vision.* and audio.* sub-configs
		// its reader does not handle. alpakka reports the GGUF values, since no client depends on them.
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

// Captured from ollama's /api/show for every local model. Recapture when a tag is
// rebuilt by posting each name from ollama list to localhost:11434/api/show.
func showGolden(t *testing.T) map[string]struct {
	Capabilities []string         `json:"capabilities"`
	Details      api.ModelDetails `json:"details"`
	Parameters   string           `json:"parameters"`
} {
	t.Helper()
	b, err := os.ReadFile("testdata/show.json")
	if err != nil {
		t.Skip("no captured show.json")
	}
	var out map[string]struct {
		Capabilities []string         `json:"capabilities"`
		Details      api.ModelDetails `json:"details"`
		Parameters   string           `json:"parameters"`
	}
	if err := json.Unmarshal(b, &out); err != nil {
		t.Fatal(err)
	}
	return out
}

// /api/show is the target rather than /api/tags, since tags answers from a pull-time
// cache that disagrees with the live computation clients query.
func TestCapabilitiesMatchOllamaShow(t *testing.T) {
	s := testStore(t)
	want := showGolden(t)

	models, err := s.List()
	if err != nil {
		t.Fatal(err)
	}
	for i := range models {
		m := &models[i]
		w, ok := want[m.Name]
		if !ok {
			continue
		}
		got := make([]string, 0, 4)
		for _, c := range m.Capabilities() {
			got = append(got, string(c))
		}
		sort.Strings(got)
		if !reflect.DeepEqual(got, w.Capabilities) {
			t.Errorf("%s: capabilities = %v, want %v", m.Name, got, w.Capabilities)
		}
	}
}

func TestManifestRelRoundTripsCanonicalName(t *testing.T) {
	cases := map[string]string{
		"qwen3:0.6b":           "registry.ollama.ai/library/qwen3/0.6b",
		"library/qwen3:latest": "registry.ollama.ai/library/qwen3/latest",
		"someone/model:v2":     "registry.ollama.ai/someone/model/v2",
		"hf.co/user/model:q4":  "hf.co/user/model/q4",
	}
	for name, want := range cases {
		got, ok := manifestRel(name)
		if !ok {
			t.Errorf("%s: no path", name)
			continue
		}
		if got != want {
			t.Errorf("%s -> %s, want %s", name, got, want)
		}
		// The path must name the model it came from, or Get would serve the wrong manifest.
		canonical, ok := canonicalName(got)
		if !ok {
			t.Errorf("%s: canonicalName rejected %s", name, got)
			continue
		}
		if name == "library/qwen3:latest" {
			// Ollama canonicalises this one to its bare form.
			continue
		}
		if canonical != name {
			t.Errorf("%s round-tripped to %s", name, canonical)
		}
	}
}

func TestManifestRelRejectsTraversal(t *testing.T) {
	for _, name := range []string{
		"../../../etc/passwd:latest",
		"..:latest",
		"a/../../b:latest",
		"qwen3",
	} {
		if rel, ok := manifestRel(name); ok {
			t.Errorf("%s accepted as %s", name, rel)
		}
	}
}
