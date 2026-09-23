package store

import "strings"

// Publishers name files <model>-<quant>.gguf, so the quant token splits the model
// name from the tag.
type GGUFName struct {
	Name      string
	Tag       string
	Projector bool
	Draft     bool
	Part      int
	Parts     int
}

// Unsloth's UD- marker is dropped: it names the recipe, not the quant, and would
// give one quantization two tags.
func ParseGGUFName(filename string) GGUFName {
	base := strings.TrimSuffix(lastSegment(filename), ".gguf")

	var out GGUFName
	if tag, part, ok := splitPart(base); ok {
		out.Part, out.Parts = part, partsOf(base)
		base = tag
	}
	if strings.Contains(strings.ToLower(base), "mmproj") {
		out.Projector = true
		return out
	}
	if isDraftName(base) {
		out.Draft = true
		return out
	}

	segs := strings.Split(base, "-")
	quant := -1
	for i, s := range segs {
		if isQuantToken(s) {
			quant = i
			break
		}
	}
	if quant < 0 {
		out.Name = slug(base)
		return out
	}

	var name []string
	for _, s := range segs[:quant] {
		if strings.EqualFold(s, "UD") {
			continue
		}
		name = append(name, s)
	}
	out.Name = slug(strings.Join(name, "-"))
	out.Tag = slug(strings.Join(segs[quant:], "-"))
	return out
}

// Only "eagle" and "draft" qualify, since "mtp" would match qwen3.5-9b-mtp, a whole
// model with a built-in MTP head.
func isDraftName(base string) bool {
	l := strings.ToLower(base)
	return strings.Contains(l, "eagle") || strings.Contains(l, "draft")
}

func Slug(s string) string { return slug(s) }

func slug(s string) string {
	s = strings.ToLower(strings.Trim(s, "-_"))
	s = strings.ReplaceAll(s, "_", "-")
	for strings.Contains(s, "--") {
		s = strings.ReplaceAll(s, "--", "-")
	}
	return s
}

func isQuantToken(s string) bool {
	u := strings.ToUpper(s)
	switch u {
	case "BF16", "F16", "F32", "FP16", "FP32":
		return true
	}
	switch {
	case strings.HasPrefix(u, "IQ"), strings.HasPrefix(u, "TQ"):
		return len(u) > 2 && u[2] >= '0' && u[2] <= '9'
	case strings.HasPrefix(u, "Q"):
		return len(u) > 1 && u[1] >= '0' && u[1] <= '9'
	}
	return false
}

func partsOf(base string) int {
	tail := base[len(base)-partSuffix:]
	n := 0
	for _, r := range tail[1+partWidth+len(partMarker):] {
		n = n*10 + int(r-'0')
	}
	return n
}

func lastSegment(p string) string {
	if i := strings.LastIndexAny(p, `/\`); i >= 0 {
		return p[i+1:]
	}
	return p
}
