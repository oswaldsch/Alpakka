package gguf

import (
	"os"
	"testing"
)

// The Q3_K_M blob of qwen3.8-27b-q3-32k. Ollama reports family qwen35,
// 27.3B parameters, Q3_K_M, context 262144, embedding 5120 for this file.
const qwen3Blob = "/var/lib/ollama/.ollama/models/blobs/" +
	"sha256-7f3b845b563888ec3abc269474cf744bf703a7ce8766dbb7f696c63975facfd7"

func TestOpenRealBlob(t *testing.T) {
	if _, err := os.Stat(qwen3Blob); err != nil {
		t.Skip("model blob not present")
	}
	f, err := Open(qwen3Blob)
	if err != nil {
		t.Fatal(err)
	}
	if got := f.Architecture(); got != "qwen35" {
		t.Errorf("architecture = %q, want qwen35", got)
	}
	if len(f.Tensors) != int(f.TensorCount) || f.TensorCount == 0 {
		t.Fatalf("read %d tensor infos, header says %d", len(f.Tensors), f.TensorCount)
	}
	if got := HumanParams(f.ParameterCount()); got != "27.3B" {
		t.Errorf("parameter_size = %q, want 27.3B", got)
	}
	ft, ok := f.Uint("general.file_type")
	if !ok {
		t.Fatal("no general.file_type")
	}
	if got := FileTypeName(ft); got != "Q3_K_M" {
		t.Errorf("quantization_level = %q, want Q3_K_M", got)
	}
	if got, _ := f.ArchUint("context_length"); got != 262144 {
		t.Errorf("context_length = %d, want 262144", got)
	}
	if got, _ := f.ArchUint("embedding_length"); got != 5120 {
		t.Errorf("embedding_length = %d, want 5120", got)
	}
	if tmpl, ok := f.String("tokenizer.chat_template"); !ok || len(tmpl) == 0 {
		t.Error("no tokenizer.chat_template")
	}
}

func TestHumanParams(t *testing.T) {
	for _, c := range []struct {
		n    uint64
		want string
	}{
		{27_300_000_000, "27.3B"},
		{24_000_000_000, "24B"},
		{596_049_920, "596M"},
		{0, ""},
	} {
		if got := HumanParams(c.n); got != c.want {
			t.Errorf("HumanParams(%d) = %q, want %q", c.n, got, c.want)
		}
	}
}
