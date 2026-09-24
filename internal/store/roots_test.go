package store

import (
	"errors"
	"path/filepath"
	"strings"
	"testing"

	"github.com/oswaldsch/alpakka/internal/gguf/gguftest"
)

func layered(t *testing.T) *Store {
	t.Helper()
	first, second := t.TempDir(), t.TempDir()
	kv := map[string]any{"general.architecture": "qwen35", "general.file_type": uint32(15)}

	gguftest.Write(t, filepath.Join(first, "shared", "q4-k-m.gguf"), kv)
	gguftest.Write(t, filepath.Join(first, "only-first", "q8-0.gguf"), kv)
	gguftest.Write(t, filepath.Join(second, "shared", "q4-k-m.gguf"), kv)
	gguftest.Write(t, filepath.Join(second, "only-second", "q8-0.gguf"), kv)

	return New(nil, first, second)
}

func TestMultiFirstRootWins(t *testing.T) {
	first, second := t.TempDir(), t.TempDir()
	kv := map[string]any{"general.architecture": "qwen35", "general.file_type": uint32(15)}
	gguftest.Write(t, filepath.Join(first, "shared", "q4-k-m.gguf"), kv)
	gguftest.Write(t, filepath.Join(second, "shared", "q4-k-m.gguf"), kv)

	var shadowed int
	m := New(func(string, ...any) { shadowed++ }, first, second)

	models, err := m.List()
	if err != nil {
		t.Fatal(err)
	}
	if len(models) != 1 {
		t.Fatalf("List = %v, want one entry", names(models))
	}
	if !strings.HasPrefix(models[0].ModelPath, first) {
		t.Errorf("ModelPath = %s, want it under %s", models[0].ModelPath, first)
	}
	if shadowed != 1 {
		t.Errorf("logged %d shadowed models, want 1", shadowed)
	}

	got, err := m.Get("shared:q4-k-m")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(got.ModelPath, first) {
		t.Errorf("Get returned %s", got.ModelPath)
	}
}

func TestMultiLogsAShadowedNameOnce(t *testing.T) {
	m := layered(t)
	var shadowed int
	m.logf = func(string, ...any) { shadowed++ }
	for range 3 {
		if _, err := m.List(); err != nil {
			t.Fatal(err)
		}
	}
	if shadowed != 1 {
		t.Errorf("logged %d times, want 1", shadowed)
	}
}

func TestMultiListsEveryRoot(t *testing.T) {
	m := layered(t)
	models, err := m.List()
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"only-first:q8-0", "only-second:q8-0", "shared:q4-k-m"}
	if got := names(models); !equal(got, want) {
		t.Errorf("List = %v, want %v", got, want)
	}
}

func TestMultiGetFallsThroughToLaterRoots(t *testing.T) {
	m := layered(t)
	if _, err := m.Get("only-second:q8-0"); err != nil {
		t.Errorf("Get(only-second) = %v", err)
	}
	_, err := m.Get("absent:q8-0")
	if !errors.Is(err, ErrNotFound) {
		t.Errorf("Get(absent) = %v, want ErrNotFound", err)
	}
}

func TestMultiKeepsTheAmbiguousNameError(t *testing.T) {
	first, second := t.TempDir(), t.TempDir()
	kv := map[string]any{"general.architecture": "qwen35", "general.file_type": uint32(15)}
	gguftest.Write(t, filepath.Join(first, "qwen", "q4-k-m.gguf"), kv)
	gguftest.Write(t, filepath.Join(first, "qwen", "q8-0.gguf"), kv)
	gguftest.Write(t, filepath.Join(second, "qwen", "iq3-xxs.gguf"), kv)

	m := New(nil, first, second)
	_, err := m.Get("qwen")
	if err == nil {
		t.Fatal("expected an error")
	}
	if errors.Is(err, ErrNotFound) || !strings.Contains(err.Error(), "q8-0") {
		t.Errorf("error = %v, want the tags of the first root", err)
	}
}

func TestMultiSurvivesAnUnreadableRoot(t *testing.T) {
	good := t.TempDir()
	gguftest.Write(t, filepath.Join(good, "fine", "q8-0.gguf"), map[string]any{
		"general.architecture": "qwen35",
	})
	m := New(nil, filepath.Join(t.TempDir(), "gone"), good)

	models, err := m.List()
	if err != nil {
		t.Fatal(err)
	}
	if got := names(models); !equal(got, []string{"fine:q8-0"}) {
		t.Errorf("List = %v", got)
	}
}
