package store

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// fakeSource stands in for an ollama store so the test does not need one
// pulled on the machine.
type fakeSource []Model

func (f fakeSource) List() ([]Model, error) { return f, nil }

func (f fakeSource) Get(name string) (*Model, error) {
	for i := range f {
		if f[i].Name == name {
			return &f[i], nil
		}
	}
	return nil, ErrNotFound
}

func blob(t *testing.T, dir, name string) string {
	t.Helper()
	p := filepath.Join(dir, name)
	if err := os.WriteFile(p, []byte("GGUF"+name), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

func importFixture(t *testing.T) (fakeSource, string) {
	t.Helper()
	blobs := t.TempDir()
	return fakeSource{
		{
			Name:          "qwen3.8-27b-q3-32k:latest",
			ModelPath:     blob(t, blobs, "weights"),
			ProjectorPath: blob(t, blobs, "projector"),
		},
		{
			Name:      "hf.co/unsloth/Qwen3.8-27B-GGUF:Q3_K_M",
			ModelPath: blob(t, blobs, "hf-weights"),
		},
	}, t.TempDir()
}

func TestPlanImportNamesAndPaths(t *testing.T) {
	src, dest := importFixture(t)
	actions, err := PlanImport(src, dest)
	if err != nil {
		t.Fatal(err)
	}
	if len(actions) != 2 {
		t.Fatalf("planned %d actions", len(actions))
	}

	if actions[0].Name != "qwen3.8-27b-q3-32k:latest" {
		t.Errorf("name = %q", actions[0].Name)
	}
	want := []string{
		filepath.Join(dest, "qwen3.8-27b-q3-32k", "latest.gguf"),
		filepath.Join(dest, "qwen3.8-27b-q3-32k", "latest.mmproj.gguf"),
	}
	for i, l := range actions[0].Links {
		if l.To != want[i] {
			t.Errorf("link %d to %q, want %q", i, l.To, want[i])
		}
	}

	// A namespaced name nests, and the tag is slugged like any other.
	if got := actions[1].Links[0].To; got != filepath.Join(
		dest, "hf.co", "unsloth", "qwen3.8-27b-gguf", "q3-k-m.gguf") {
		t.Errorf("hf link to %q", got)
	}
}

// The same weights must land on the same tag whether they were pulled or
// imported, so an ollama tag goes through the pull slug rules too.
func TestImportTag(t *testing.T) {
	for in, want := range map[string]string{
		"UD-Q4_K_M": "q4-k-m",
		"Q3_K_M":    "q3-k-m",
		"IQ4_XS":    "iq4-xs",
		"latest":    "latest",
		"9b":        "9b",
		"ud-coder":  "ud-coder",
	} {
		if got := importTag(in); got != want {
			t.Errorf("importTag(%q) = %q, want %q", in, got, want)
		}
	}
}

// Dry run is the default, so planning must not touch the destination.
func TestPlanImportWritesNothing(t *testing.T) {
	src, dest := importFixture(t)
	if _, err := PlanImport(src, dest); err != nil {
		t.Fatal(err)
	}
	entries, err := os.ReadDir(dest)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 0 {
		t.Errorf("planning created %d entries in %s", len(entries), dest)
	}
}

func TestImportLinksRatherThanCopies(t *testing.T) {
	src, dest := importFixture(t)
	actions, err := PlanImport(src, dest)
	if err != nil {
		t.Fatal(err)
	}
	for _, a := range actions {
		if err := a.Link(); err != nil {
			t.Fatal(err)
		}
	}

	for _, a := range actions {
		for _, l := range a.Links {
			from, err := os.Stat(l.From)
			if err != nil {
				t.Fatal(err)
			}
			to, err := os.Stat(l.To)
			if err != nil {
				t.Fatal(err)
			}
			if !os.SameFile(from, to) {
				t.Errorf("%s is a copy of %s, not a link", l.To, l.From)
			}
		}
	}

	// Re-running is a no-op rather than a second set of links.
	again, err := PlanImport(src, dest)
	if err != nil {
		t.Fatal(err)
	}
	for _, a := range again {
		if a.Skip != "already linked" {
			t.Errorf("%s: skip = %q, want \"already linked\"", a.Model, a.Skip)
		}
	}
}

// A different file under the target name belongs to somebody else and the blob
// it would replace cannot be recovered, so it is refused rather than clobbered.
func TestImportRefusesADifferentFile(t *testing.T) {
	src, dest := importFixture(t)
	occupied := filepath.Join(dest, "qwen3.8-27b-q3-32k")
	if err := os.MkdirAll(occupied, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(occupied, "latest.gguf"), []byte("mine"), 0o644); err != nil {
		t.Fatal(err)
	}

	actions, err := PlanImport(src, dest)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(actions[0].Skip, "different file") {
		t.Errorf("skip = %q", actions[0].Skip)
	}
	if len(actions[0].Links) != 0 {
		t.Errorf("planned %d links anyway", len(actions[0].Links))
	}
}

// What is imported has to be what the directory store then serves.
func TestImportedModelsReadBack(t *testing.T) {
	blobs, dest := t.TempDir(), t.TempDir()
	weights := filepath.Join(blobs, "weights")
	if err := os.WriteFile(weights, ggufBytes(map[string]any{
		"general.architecture": "qwen35",
		"general.file_type":    uint32(12),
	}), 0o644); err != nil {
		t.Fatal(err)
	}

	src := fakeSource{{Name: "qwen3.8-27b-q3-32k:latest", ModelPath: weights}}
	actions, err := PlanImport(src, dest)
	if err != nil {
		t.Fatal(err)
	}
	if err := actions[0].Link(); err != nil {
		t.Fatal(err)
	}

	m, err := NewDir(dest).Get("qwen3.8-27b-q3-32k")
	if err != nil {
		t.Fatal(err)
	}
	if m.Name != "qwen3.8-27b-q3-32k:latest" || m.Details().QuantizationLevel != "Q3_K_M" {
		t.Errorf("model = %s, details = %+v", m.Name, m.Details())
	}
}
