package store

import (
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"github.com/ollama/ollama/types/model"
)

const toolTemplate = `{% for m in messages %}{{ m.content }}{% endfor %}` +
	`{% if tools %}{{ tools }}{% endif %}<think></think>`

func dirFixture(t *testing.T) *DirStore {
	t.Helper()
	root := t.TempDir()

	writeGGUF(t, filepath.Join(root, "qwen3.8-27b", "iq3-xxs.gguf"), map[string]any{
		"general.architecture":    "qwen35",
		"general.file_type":       uint32(23),
		"qwen35.context_length":   uint32(262144),
		"qwen35.embedding_length": uint32(5120),
		"tokenizer.chat_template": toolTemplate,
	})
	writeGGUF(t, filepath.Join(root, "qwen3.8-27b", "iq4-xs.gguf"), map[string]any{
		"general.architecture":    "qwen35",
		"general.file_type":       uint32(30),
		"qwen35.context_length":   uint32(262144),
		"tokenizer.chat_template": toolTemplate,
	})
	writeGGUF(t, filepath.Join(root, "qwen3.8-27b", "iq4-xs.mmproj.gguf"), map[string]any{
		"general.architecture": "clip",
	})
	writeGGUF(t, filepath.Join(root, "qwen3.8-27b", "imatrix_unsloth.gguf"), map[string]any{
		"general.type": "imatrix",
	})
	writeGGUF(t, filepath.Join(root, "embedgemma", "q8-0.gguf"), map[string]any{
		"general.architecture": "gemma3",
		"general.file_type":    uint32(7),
		"pooling_type":         uint32(1),
	})
	writeGGUF(t, filepath.Join(root, "unsloth", "glm-4.7-flash", "iq4-xs.gguf"), map[string]any{
		"general.architecture": "glm4",
		"general.file_type":    uint32(30),
	})
	return NewDir(root)
}

func names(models []Model) []string {
	out := make([]string, 0, len(models))
	for _, m := range models {
		out = append(out, m.Name)
	}
	sort.Strings(out)
	return out
}

func TestDirListNamesEveryTag(t *testing.T) {
	s := dirFixture(t)
	models, err := s.List()
	if err != nil {
		t.Fatal(err)
	}
	want := []string{
		"embedgemma:q8-0",
		"qwen3.8-27b:iq3-xxs",
		"qwen3.8-27b:iq4-xs",
		"unsloth/glm-4.7-flash:iq4-xs",
	}
	if got := names(models); !equal(got, want) {
		t.Errorf("List = %v, want %v", got, want)
	}
}

func TestDirListSortedNewestFirst(t *testing.T) {
	s := dirFixture(t)
	models, err := s.List()
	if err != nil {
		t.Fatal(err)
	}
	for i := 1; i < len(models); i++ {
		if models[i-1].ModifiedAt.Before(models[i].ModifiedAt) {
			t.Fatalf("%s is older than %s", models[i-1].Name, models[i].Name)
		}
	}
}

func TestDirGetResolvesTagAndProjector(t *testing.T) {
	s := dirFixture(t)
	m, err := s.Get("qwen3.8-27b:iq4-xs")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasSuffix(m.ModelPath, "iq4-xs.gguf") {
		t.Errorf("ModelPath = %s", m.ModelPath)
	}
	if !strings.HasSuffix(m.ProjectorPath, "iq4-xs.mmproj.gguf") {
		t.Errorf("ProjectorPath = %s", m.ProjectorPath)
	}
	if m.Template != toolTemplate {
		t.Errorf("Template = %q, want the GGUF chat template", m.Template)
	}
	// config.toml owns these for a directory-backed model.
	if m.System != "" || len(m.Params) != 0 {
		t.Errorf("System = %q, Params = %v; both should be empty", m.System, m.Params)
	}
}

func TestDirBareNameNeedsAUniqueTag(t *testing.T) {
	s := dirFixture(t)
	if _, err := s.Get("qwen3.8-27b"); err == nil {
		t.Fatal("expected an error naming the tags")
	} else if !strings.Contains(err.Error(), "iq3-xxs") || !strings.Contains(err.Error(), "iq4-xs") {
		t.Errorf("error should name the choices, got %v", err)
	}

	m, err := s.Get("embedgemma")
	if err != nil {
		t.Fatal(err)
	}
	if m.Name != "embedgemma:q8-0" {
		t.Errorf("Name = %q", m.Name)
	}
}

func TestDirDetailsComeFromTheGGUF(t *testing.T) {
	s := dirFixture(t)
	m, err := s.Get("qwen3.8-27b:iq3-xxs")
	if err != nil {
		t.Fatal(err)
	}
	d := m.Details()
	if d.Format != "gguf" || d.Family != "qwen35" || d.QuantizationLevel != "IQ3_XXS" {
		t.Errorf("details = %+v", d)
	}
	if d.ContextLength != 262144 || d.EmbeddingLength != 5120 {
		t.Errorf("lengths = %d/%d", d.ContextLength, d.EmbeddingLength)
	}
}

func TestDirCapabilities(t *testing.T) {
	s := dirFixture(t)
	for _, c := range []struct {
		name string
		want []string
	}{
		{"qwen3.8-27b:iq3-xxs", []string{"completion", "thinking", "tools"}},
		{"qwen3.8-27b:iq4-xs", []string{"completion", "thinking", "tools", "vision"}},
		{"embedgemma:q8-0", []string{"embedding"}},
		{"unsloth/glm-4.7-flash:iq4-xs", []string{"completion"}},
	} {
		m, err := s.Get(c.name)
		if err != nil {
			t.Fatal(err)
		}
		got := make([]string, 0, 4)
		for _, cap := range m.Capabilities() {
			got = append(got, string(cap))
		}
		sort.Strings(got)
		if !equal(got, c.want) {
			t.Errorf("%s: capabilities = %v, want %v", c.name, got, c.want)
		}
	}
}

func TestDirEmbeddingModelIsDetected(t *testing.T) {
	s := dirFixture(t)
	m, err := s.Get("embedgemma:q8-0")
	if err != nil {
		t.Fatal(err)
	}
	if !m.IsEmbedding() || !m.DeclaresPooling() {
		t.Errorf("%s should be an embedding model", m.Name)
	}
	for _, c := range m.Capabilities() {
		if c == model.CapabilityCompletion {
			t.Error("an embedding model should not claim completion")
		}
	}
}

func TestDirDigestIsStableAndDistinct(t *testing.T) {
	s := dirFixture(t)
	first, err := s.Get("qwen3.8-27b:iq3-xxs")
	if err != nil {
		t.Fatal(err)
	}
	again, err := NewDir(s.Root()).Get("qwen3.8-27b:iq3-xxs")
	if err != nil {
		t.Fatal(err)
	}
	if first.Digest != again.Digest {
		t.Errorf("digest changed between opens: %s vs %s", first.Digest, again.Digest)
	}
	other, err := s.Get("qwen3.8-27b:iq4-xs")
	if err != nil {
		t.Fatal(err)
	}
	if first.Digest == other.Digest {
		t.Error("two quants share a digest")
	}
	if first.Digest == "" || first.Size == 0 {
		t.Errorf("digest = %q, size = %d", first.Digest, first.Size)
	}
}

func TestDirSplitModelIsOneTag(t *testing.T) {
	root := t.TempDir()
	kv := map[string]any{"general.architecture": "qwen35", "general.file_type": uint32(15)}
	writeGGUF(t, filepath.Join(root, "big", "q4-k-m-00001-of-00003.gguf"), kv)
	writeGGUF(t, filepath.Join(root, "big", "q4-k-m-00002-of-00003.gguf"), kv)
	writeGGUF(t, filepath.Join(root, "big", "q4-k-m-00003-of-00003.gguf"), kv)

	s := NewDir(root)
	models, err := s.List()
	if err != nil {
		t.Fatal(err)
	}
	if len(models) != 1 || models[0].Name != "big:q4-k-m" {
		t.Fatalf("List = %v", names(models))
	}
	if !strings.HasSuffix(models[0].ModelPath, "q4-k-m-00001-of-00003.gguf") {
		t.Errorf("ModelPath = %s, want the first part", models[0].ModelPath)
	}
	// llama.cpp loads every part, so /api/tags has to report all of them.
	if want := 3 * int64(len(ggufBytes(kv))); models[0].Size != want {
		t.Errorf("Size = %d, want %d", models[0].Size, want)
	}
}

func TestDirSkipsUnparseableFiles(t *testing.T) {
	root := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, "broken"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "broken", "half.gguf"), []byte("GGU"), 0o644); err != nil {
		t.Fatal(err)
	}
	writeGGUF(t, filepath.Join(root, "fine", "q8-0.gguf"), map[string]any{
		"general.architecture": "qwen35",
	})

	models, err := NewDir(root).List()
	if err != nil {
		t.Fatal(err)
	}
	if got := names(models); !equal(got, []string{"fine:q8-0"}) {
		t.Errorf("List = %v", got)
	}
}

func TestDirIgnoresOrphanProjector(t *testing.T) {
	root := t.TempDir()
	writeGGUF(t, filepath.Join(root, "orphan", "q4-k-m.mmproj.gguf"), map[string]any{
		"general.architecture": "clip",
	})
	models, err := NewDir(root).List()
	if err != nil {
		t.Fatal(err)
	}
	if len(models) != 0 {
		t.Errorf("List = %v", names(models))
	}
}

func TestDirMissingRootListsNothing(t *testing.T) {
	models, err := NewDir(filepath.Join(t.TempDir(), "absent")).List()
	if err != nil {
		t.Fatal(err)
	}
	if len(models) != 0 {
		t.Errorf("List = %v", names(models))
	}
}

func TestDirGetRejectsTraversal(t *testing.T) {
	s := dirFixture(t)
	for _, name := range []string{
		"../../../etc/passwd:latest",
		"..:latest",
		"qwen3.8-27b/..:iq3-xxs",
		":iq3-xxs",
	} {
		if m, err := s.Get(name); err == nil {
			t.Errorf("%s resolved to %s", name, m.ModelPath)
		}
	}
}

func TestSplitPart(t *testing.T) {
	for _, c := range []struct {
		base string
		tag  string
		part int
		ok   bool
	}{
		{"q4-k-m-00001-of-00003", "q4-k-m", 1, true},
		{"iq3-xxs-00012-of-00012", "iq3-xxs", 12, true},
		{"q4-k-m", "", 0, false},
		{"q4-k-m-0001-of-0003", "", 0, false},
		{"q4-k-m-00001-to-00003", "", 0, false},
		{"-00001-of-00003", "", 0, false},
	} {
		tag, part, ok := splitPart(c.base)
		if ok != c.ok || tag != c.tag || part != c.part {
			t.Errorf("splitPart(%q) = %q, %d, %v; want %q, %d, %v",
				c.base, tag, part, ok, c.tag, c.part, c.ok)
		}
	}
}

func equal(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
