package main

import (
	"io"
	"log"
	"path/filepath"
	"strings"
	"testing"

	"github.com/oswaldsch/alpakka/internal/config"
	"github.com/oswaldsch/alpakka/internal/hub"
)

func discardLogger() *log.Logger { return log.New(io.Discard, "", 0) }

func TestRunRejectsUnknownCommand(t *testing.T) {
	err := run([]string{"frobnicate"})
	if err == nil || !strings.Contains(err.Error(), `unknown command "frobnicate"`) {
		t.Fatalf("got %v, want an unknown command error", err)
	}
}

func TestWritableRoot(t *testing.T) {
	first, second := t.TempDir(), t.TempDir()

	tests := []struct {
		name     string
		roots    []string
		override string
		want     string
		wantErr  bool
	}{
		{name: "override wins", roots: []string{first}, override: "/elsewhere", want: "/elsewhere"},
		{name: "first root", roots: []string{first, second}, want: first},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := config.Config{Store: config.Store{Roots: tt.roots}}
			got, err := writableRoot(cfg, tt.override)
			if (err != nil) != tt.wantErr {
				t.Fatalf("err = %v, wantErr %t", err, tt.wantErr)
			}
			if got != tt.want {
				t.Fatalf("got %q, want %q", got, tt.want)
			}
		})
	}
}

func TestWantProjector(t *testing.T) {
	plan := &hub.Plan{Repo: "owner/repo", Projector: &hub.Entry{Path: "mmproj-F16.gguf"}}

	tests := []struct {
		mode    string
		want    bool
		wantErr bool
	}{
		{mode: "yes", want: true},
		{mode: "TRUE", want: true},
		{mode: "no", want: false},
		{mode: "false", want: false},
		{mode: "maybe", wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.mode, func(t *testing.T) {
			got, err := wantProjector(tt.mode, plan, discardLogger())
			if (err != nil) != tt.wantErr {
				t.Fatalf("err = %v, wantErr %t", err, tt.wantErr)
			}
			if got != tt.want {
				t.Fatalf("got %t, want %t", got, tt.want)
			}
		})
	}
}

func TestOpenRoots(t *testing.T) {
	plain, other := t.TempDir(), t.TempDir()
	missing := filepath.Join(t.TempDir(), "absent")

	t.Run("skips a missing root", func(t *testing.T) {
		cfg := config.Config{Store: config.Store{Roots: []string{missing, plain}}}
		roots, err := openRoots(cfg, "", discardLogger())
		if err != nil {
			t.Fatal(err)
		}
		if len(roots) != 1 || roots[0] != plain {
			t.Fatalf("got %v, want only %s", roots, plain)
		}
	})

	t.Run("override replaces the configured roots", func(t *testing.T) {
		cfg := config.Config{Store: config.Store{Roots: []string{plain}}}
		roots, err := openRoots(cfg, other, discardLogger())
		if err != nil {
			t.Fatal(err)
		}
		if len(roots) != 1 || roots[0] != other {
			t.Fatalf("got %v, want only the override root", roots)
		}
	})

	t.Run("no readable root is an error", func(t *testing.T) {
		cfg := config.Config{Store: config.Store{Roots: []string{missing}}}
		if _, err := openRoots(cfg, "", discardLogger()); err == nil {
			t.Fatal("want an error when no root exists")
		}
	})
}
