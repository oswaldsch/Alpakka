package store

import (
	"encoding/json"
	"os"
	"reflect"
	"sort"
	"testing"

	"github.com/ollama/ollama/api"
)

// testStore opens the real ollama store, skipping when it is not usable.
//
// An empty store is a skip, not a failure: the directory exists on any machine
// that has ever run ollama, and these tests compare against models that have
// actually been pulled.
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
//
// It only means anything on the machine tags.json was captured from. Elsewhere
// the comparison is against another machine's store, where a differing count is
// expected and modified_at — the manifest's mtime, set when the model was
// pulled here — can never match. Skipping says that plainly instead of failing
// for a reason that has nothing to do with the code.
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

// showGolden is /api/show captured from the real ollama for every local model.
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

// TestCapabilitiesMatchOllamaShow checks capability detection against ollama's
// live computation. /api/show is the target rather than /api/tags: tags answers
// from a cache built at pull time and the two disagree in ollama itself, while
// show is what clients query before deciding to send tools or request thinking.
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

// manifestRel is what makes Get a single open instead of a walk over the whole
// store, so it has to agree with canonicalName in both directions.
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
		// The path must name the model it came from, or Get would serve the
		// wrong manifest.
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

// The model name arrives from an HTTP request, so it must not be able to point
// the store outside its own directory.
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
