package store

import "strings"

// GGUFName is what a GGUF filename says about the model inside it.
//
// Publishers name files <model>-<quant>.gguf, so the quant token is the seam:
// everything before it names the model and everything from it names the tag.
type GGUFName struct {
	Name      string
	Tag       string
	Projector bool
	Part      int // split part number, zero when the file is not split
	Parts     int
}

// ParseGGUFName splits a published GGUF filename into a model name and tag.
//
// Unsloth's UD- marker is dropped: it says the quant was made with their
// dynamic recipe, not which quant it is, and keeping it would give the same
// quantization two different tags depending on who built it.
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

// Slug is the lowercase, hyphenated form a name or tag is stored under.
func Slug(s string) string { return slug(s) }

func slug(s string) string {
	s = strings.ToLower(strings.Trim(s, "-_"))
	s = strings.ReplaceAll(s, "_", "-")
	for strings.Contains(s, "--") {
		s = strings.ReplaceAll(s, "--", "-")
	}
	return s
}

// isQuantToken recognises the llama.cpp quantization names as they appear in a
// filename segment: Q8_0, Q4_K_M, IQ3_XXS, TQ1_0, BF16 and the float types.
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

// partsOf reads the total from a "-00002-of-00003" suffix already matched by
// splitPart.
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
