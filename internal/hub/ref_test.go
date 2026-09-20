package hub

import "testing"

func TestParseRef(t *testing.T) {
	const repo = "unsloth/Qwen3.8-27B-GGUF"
	const file = "Qwen3.8-27B-UD-IQ3_XXS.gguf"

	for _, c := range []struct {
		in   string
		want Ref
	}{
		{"https://huggingface.co/" + repo + "/blob/main/" + file,
			Ref{Repo: repo, Revision: "main", File: file}},
		{"https://huggingface.co/" + repo + "/resolve/main/" + file,
			Ref{Repo: repo, Revision: "main", File: file}},
		{"hf.co/" + repo + "/UD-IQ3_XXS",
			Ref{Repo: repo, Revision: "main", Quant: "iq3-xxs"}},
		{repo + ":UD-IQ3_XXS",
			Ref{Repo: repo, Revision: "main", Quant: "iq3-xxs"}},

		{"http://huggingface.co/" + repo + "/resolve/main/" + file,
			Ref{Repo: repo, Revision: "main", File: file}},
		{"huggingface.co/" + repo + "/IQ4_XS",
			Ref{Repo: repo, Revision: "main", Quant: "iq4-xs"}},
		{repo + ":q4-k-m", Ref{Repo: repo, Revision: "main", Quant: "q4-k-m"}},
		{repo, Ref{Repo: repo, Revision: "main"}},
		{"https://huggingface.co/" + repo + "/blob/v2/UD-IQ3_XXS/" + file,
			Ref{Repo: repo, Revision: "v2", File: "UD-IQ3_XXS/" + file}},
		{"hf.co/" + repo + "/", Ref{Repo: repo, Revision: "main"}},
	} {
		got, err := ParseRef(c.in)
		if err != nil {
			t.Errorf("ParseRef(%q) = %v", c.in, err)
			continue
		}
		if got != c.want {
			t.Errorf("ParseRef(%q) = %+v, want %+v", c.in, got, c.want)
		}
	}
}

func TestParseRefRejects(t *testing.T) {
	for _, in := range []string{
		"",
		"qwen3",
		"https://example.com/unsloth/repo/blob/main/x.gguf",
		"unsloth/repo/extra/path",
		"https://huggingface.co/unsloth/repo/blob/main/README.md",
	} {
		if got, err := ParseRef(in); err == nil {
			t.Errorf("ParseRef(%q) = %+v, want an error", in, got)
		}
	}
}

func TestNormalizeQuant(t *testing.T) {
	for in, want := range map[string]string{
		"UD-IQ3_XXS":   "iq3-xxs",
		"IQ3_XXS":      "iq3-xxs",
		"iq3-xxs":      "iq3-xxs",
		"Q4_K_M":       "q4-k-m",
		"UD-Q4_K_XL":   "q4-k-xl",
		"Q8_0.gguf":    "q8-0",
		"  BF16      ": "bf16",
	} {
		if got := normalizeQuant(in); got != want {
			t.Errorf("normalizeQuant(%q) = %q, want %q", in, got, want)
		}
	}
}
