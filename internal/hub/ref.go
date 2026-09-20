// Package hub pulls GGUF models from HuggingFace into alpakka's model store.
package hub

import (
	"fmt"
	"strings"
)

// Ref is a reference to something to pull.
//
// Either File names one exact file in the repo, or Quant selects among the
// repo's GGUFs by quantization. Both empty means the repo has to hold exactly
// one model for the reference to be unambiguous.
type Ref struct {
	Repo     string
	Revision string
	File     string
	Quant    string
}

const defaultRevision = "main"

// ParseRef accepts the shapes a GGUF gets copied out of a browser as:
//
//	https://huggingface.co/<repo>/blob/main/<file>.gguf
//	https://huggingface.co/<repo>/resolve/main/<file>.gguf
//	hf.co/<repo>/<quant>
//	<repo>:<quant>
func ParseRef(s string) (Ref, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return Ref{}, fmt.Errorf("empty model reference")
	}
	s = strings.TrimSuffix(s, "/")

	rest := s
	for _, prefix := range []string{"https://", "http://"} {
		rest = strings.TrimPrefix(rest, prefix)
	}
	hosted := false
	for _, host := range []string{"huggingface.co/", "hf.co/"} {
		if after, ok := strings.CutPrefix(rest, host); ok {
			rest, hosted = after, true
			break
		}
	}

	parts := strings.Split(rest, "/")
	if len(parts) < 2 || parts[0] == "" || parts[1] == "" {
		return Ref{}, fmt.Errorf("%q: expected owner/repo, optionally with a quant", s)
	}

	ref := Ref{Revision: defaultRevision}
	ref.Repo = parts[0] + "/" + parts[1]
	tail := parts[2:]

	// A tag may be written against the repo in any of the forms.
	if repo, quant, ok := strings.Cut(ref.Repo, ":"); ok {
		ref.Repo, ref.Quant = repo, normalizeQuant(quant)
	}

	switch {
	case len(tail) == 0:
	case (tail[0] == "blob" || tail[0] == "resolve") && len(tail) >= 3:
		ref.Revision = tail[1]
		ref.File = strings.Join(tail[2:], "/")
	case hosted && len(tail) == 1 && ref.Quant == "":
		ref.Quant = normalizeQuant(tail[0])
	default:
		return Ref{}, fmt.Errorf("%q: not a HuggingFace model reference", s)
	}

	if ref.File != "" && !strings.HasSuffix(ref.File, ".gguf") {
		return Ref{}, fmt.Errorf("%q: not a GGUF file", ref.File)
	}
	return ref, nil
}

// normalizeQuant reduces a quant as written anywhere to the tag form alpakka
// stores it under, so "UD-IQ3_XXS", "IQ3_XXS" and "iq3-xxs" all select one
// model.
func normalizeQuant(s string) string {
	s = strings.TrimSuffix(strings.ToLower(strings.TrimSpace(s)), ".gguf")
	s = strings.ReplaceAll(s, "_", "-")
	s = strings.TrimPrefix(s, "ud-")
	return strings.Trim(s, "-")
}
