package config

import (
	"fmt"
	"reflect"
)

// alpakka's own settings ride in the same options object. Ollama ignores
// unknown keys, so a request carrying spec_type stays valid against both servers.
func Apply(p Profile, opts map[string]any) (Profile, error) {
	if len(opts) == 0 {
		return p, nil
	}
	out := p
	fields := reflect.ValueOf(&out).Elem()

	for k, v := range opts {
		i, ok := optionFields[k]
		if !ok {
			// Ignored, as ollama ignores ours.
			continue
		}
		if err := setOption(fields.Field(i).Addr().Interface(), k, v); err != nil {
			return p, err
		}
	}

	if err := out.Validate(); err != nil {
		return p, err
	}
	return out, nil
}

var optionFields = func() map[string]int {
	t := reflect.TypeFor[Profile]()
	out := make(map[string]int, t.NumField())
	for i := range t.NumField() {
		f := t.Field(i)
		if key := f.Tag.Get("toml"); key != "" && f.Tag.Get("option") != "-" {
			out[key] = i
		}
	}
	return out
}()

func setOption(dst any, key string, v any) error {
	switch d := dst.(type) {
	case **int:
		return setInt(d, key, v)
	case **float32:
		return setFloat(d, key, v)
	case **string:
		return setStr(d, key, v)
	case **bool:
		return setBool(d, key, v)
	case *[]string:
		return setStrings(d, key, v)
	case *StringList:
		if str, ok := v.(string); ok {
			v = splitList(str)
		}
		return setStrings((*[]string)(d), key, v)
	}
	panic(fmt.Sprintf("config: option %q has no setter for %T", key, dst))
}

// JSON numbers decode as float64.
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
