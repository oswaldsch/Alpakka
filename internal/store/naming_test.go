package store

import "testing"

func TestParseGGUFName(t *testing.T) {
	for _, c := range []struct {
		file      string
		name      string
		tag       string
		projector bool
		draft     bool
		imatrix   bool
		part      int
		parts     int
	}{
		{file: "Qwen3.8-27B-UD-IQ3_XXS.gguf", name: "qwen3.8-27b", tag: "iq3-xxs"},
		{file: "Qwen3.8-27B-UD-Q4_K_XL.gguf", name: "qwen3.8-27b", tag: "q4-k-xl"},
		{file: "Qwen3.8-27B-Q8_0.gguf", name: "qwen3.8-27b", tag: "q8-0"},
		{file: "Qwen3.5-9B-Q4_K_M.gguf", name: "qwen3.5-9b", tag: "q4-k-m"},
		{file: "GLM-4.7-Flash-IQ4_XS.gguf", name: "glm-4.7-flash", tag: "iq4-xs"},
		{file: "mmproj-BF16.gguf", projector: true},
		{file: "eagle3-gpt-oss-20b-Q8_0.gguf", draft: true},
		{file: "eagle3-q8-0.gguf", draft: true},
		{file: "imatrix_unsloth.gguf", imatrix: true},

		{file: "mmproj-F16.gguf", projector: true},
		{file: "Qwen3-VL-8B-mmproj-F32.gguf", projector: true},
		{file: "UD-IQ3_XXS/Qwen3.8-27B-UD-IQ3_XXS.gguf", name: "qwen3.8-27b", tag: "iq3-xxs"},
		{file: "Qwen3.8-27B-UD-IQ3_XXS-00002-of-00003.gguf",
			name: "qwen3.8-27b", tag: "iq3-xxs", part: 2, parts: 3},
		{file: "GLM-4.7-Flash-BF16.gguf", name: "glm-4.7-flash", tag: "bf16"},
		{file: "some-model.gguf", name: "some-model"},
	} {
		got := ParseGGUFName(c.file)
		if got.Imatrix != c.imatrix {
			t.Errorf("%s: Imatrix = %v, want %v", c.file, got.Imatrix, c.imatrix)
		}
		if got.Draft != c.draft {
			t.Errorf("%s: Draft = %v, want %v", c.file, got.Draft, c.draft)
		}
		if got.Name != c.name || got.Tag != c.tag || got.Projector != c.projector ||
			got.Part != c.part || got.Parts != c.parts {
			t.Errorf("ParseGGUFName(%q) = %+v, want name=%q tag=%q projector=%v part=%d/%d",
				c.file, got, c.name, c.tag, c.projector, c.part, c.parts)
		}
	}
}

func TestIsQuantTokenIgnoresModelNames(t *testing.T) {
	for _, s := range []string{"Qwen3.8", "27B", "Flash", "GLM", "4.7", "Coder", "QwQ"} {
		if isQuantToken(s) {
			t.Errorf("%q read as a quant", s)
		}
	}
	for _, s := range []string{"Q8_0", "Q4_K_M", "IQ3_XXS", "TQ1_0", "BF16", "F16"} {
		if !isQuantToken(s) {
			t.Errorf("%q not read as a quant", s)
		}
	}
}
