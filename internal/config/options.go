package config

import (
	"fmt"
)

// Apply layers an ollama options object onto a profile.
//
// alpakka's own settings ride in the same options object as ollama's. Ollama
// ignores keys it does not recognise, so a request carrying spec_type stays
// valid against both servers, which is what keeps the extension honest.
func Apply(p Profile, opts map[string]any) (Profile, error) {
	if len(opts) == 0 {
		return p, nil
	}
	out := p

	for k, v := range opts {
		var err error
		switch k {
		// Process-level.
		case "num_ctx":
			err = setInt(&out.NumCtx, k, v)
		case "cache_type_k":
			err = setStr(&out.CacheTypeK, k, v)
		case "cache_type_v":
			err = setStr(&out.CacheTypeV, k, v)
		case "spec_type":
			err = setStr(&out.SpecType, k, v)
		case "spec_draft_n_max":
			err = setInt(&out.SpecDraftNMax, k, v)
		case "spec_draft_n_min":
			err = setInt(&out.SpecDraftNMin, k, v)
		case "num_gpu":
			err = setInt(&out.NumGPU, k, v)
		case "flash_attn":
			err = setStr(&out.FlashAttn, k, v)
		case "backend":
			err = setStr(&out.Backend, k, v)
		case "projector":
			err = setBool(&out.Projector, k, v)

		// Request-level.
		case "reasoning_effort":
			err = setStr(&out.ReasoningEffort, k, v)
		case "temperature":
			err = setFloat(&out.Temperature, k, v)
		case "top_k":
			err = setInt(&out.TopK, k, v)
		case "top_p":
			err = setFloat(&out.TopP, k, v)
		case "min_p":
			err = setFloat(&out.MinP, k, v)
		case "repeat_penalty":
			err = setFloat(&out.RepeatPenalty, k, v)
		case "seed":
			err = setInt(&out.Seed, k, v)
		case "num_predict":
			err = setInt(&out.NumPredict, k, v)
		case "stop":
			err = setStrings(&out.Stop, k, v)

		default:
			// Unknown keys are ignored, exactly as ollama ignores ours.
			continue
		}
		if err != nil {
			return p, err
		}
	}

	if err := out.Validate(); err != nil {
		return p, err
	}
	return out, nil
}

// JSON numbers decode as float64, so every numeric option arrives as one.
func setInt(dst **int, key string, v any) error {
	switch n := v.(type) {
	case float64:
		i := int(n)
		*dst = &i
	case int:
		*dst = &n
	case int64:
		i := int(n)
		*dst = &i
	default:
		return fmt.Errorf("option %q: expected a number, got %T", key, v)
	}
	return nil
}

func setFloat(dst **float32, key string, v any) error {
	switch n := v.(type) {
	case float64:
		f := float32(n)
		*dst = &f
	case float32:
		*dst = &n
	case int:
		f := float32(n)
		*dst = &f
	default:
		return fmt.Errorf("option %q: expected a number, got %T", key, v)
	}
	return nil
}

func setStr(dst **string, key string, v any) error {
	s, ok := v.(string)
	if !ok {
		return fmt.Errorf("option %q: expected a string, got %T", key, v)
	}
	*dst = &s
	return nil
}

func setBool(dst **bool, key string, v any) error {
	b, ok := v.(bool)
	if !ok {
		return fmt.Errorf("option %q: expected true or false, got %T", key, v)
	}
	*dst = &b
	return nil
}

// setStrings accepts ollama's stop option, which may be a single string or a list.
func setStrings(dst *[]string, key string, v any) error {
	switch s := v.(type) {
	case string:
		*dst = []string{s}
	case []string:
		*dst = s
	case []any:
		out := make([]string, 0, len(s))
		for _, e := range s {
			str, ok := e.(string)
			if !ok {
				return fmt.Errorf("option %q: expected strings, got %T", key, e)
			}
			out = append(out, str)
		}
		*dst = out
	default:
		return fmt.Errorf("option %q: expected a string or list, got %T", key, v)
	}
	return nil
}
