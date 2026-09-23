// Package hub pulls GGUF models from HuggingFace into alpakka's model store.
package hub

import (
	"fmt"
	"strings"
)

// Either File names one exact file or Quant selects by quantization. Both empty
// means the repo must hold exactly one model.
type Ref struct {
	Repo     string
	Revision string
	File     string
	Quant    string
}

const defaultRevision = "main"

// Accepts huggingface.co blob and resolve URLs as copied from a browser, plus hf.co/<repo>/<quant> and <repo>:<quant>.
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

// So "UD-IQ3_XXS", "IQ3_XXS" and "iq3-xxs" all select one model.
func normalizeQuant(s string) string {
	s = strings.TrimSuffix(strings.ToLower(strings.TrimSpace(s)), ".gguf")
	s = strings.ReplaceAll(s, "_", "-")
	s = strings.TrimPrefix(s, "ud-")
	return strings.Trim(s, "-")
}
