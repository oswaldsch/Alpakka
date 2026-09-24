package gguf

import "testing"

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
