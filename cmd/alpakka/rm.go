package main

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/ollama/ollama/api"

	"github.com/oswaldsch/alpakka/internal/config"
	"github.com/oswaldsch/alpakka/internal/store"
)

func rm(args []string) error {
	fs := flag.NewFlagSet("rm", flag.ExitOnError)
	var (
		configPath = fs.String("config", config.DefaultPath(), "path to config.toml")
		host       = fs.String("host", "", "server to ask whether the model is loaded, default the configured listen address")
		modelsRoot = fs.String("models", "", "delete from this model root instead of the configured ones")
		force      = fs.Bool("f", false, "delete even while the server has the model loaded")
	)
	fs.Usage = func() {
		fmt.Fprintln(fs.Output(), "alpakka rm [-f] <model>...")
		fs.PrintDefaults()
	}
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() == 0 {
		fs.Usage()
		return fmt.Errorf("rm takes at least one model")
	}

	cfg, err := config.Load(*configPath)
	if err != nil {
		return err
	}
	roots := cfg.Store.Roots
	if *modelsRoot != "" {
		roots = []string{*modelsRoot}
	}
	if len(roots) == 0 {
		roots = store.DefaultRoots()
	}
	addr := *host
	if addr == "" {
		addr = cfg.Server.Listen
	}

	s := store.New(nil, roots...)
	for _, name := range fs.Args() {
		m, err := s.Get(name)
		if err != nil {
			return err
		}
		if !*force && loadedOn(addr, m.Name) {
			return fmt.Errorf("%s is loaded: run `alpakka stop %s` first, or pass -f", m.Name, m.Name)
		}
		if err := removeModel(m); err != nil {
			return err
		}
		fmt.Printf("deleted %s (%s)\n", m.Name, humanBytes(m.Size))
		// The next root down takes the name over, which is worth knowing before it gets loaded.
		if next, err := s.Get(m.Name); err == nil {
			fmt.Printf("%s is still served, from %s\n", next.Name, filepath.Dir(next.ModelPath))
		}
	}
	return nil
}

// A server that cannot be reached has nothing loaded to break.
func loadedOn(addr, model string) bool {
	var resp api.ProcessResponse
	if err := getJSON(addr, "/api/ps", &resp); err != nil {
		return false
	}
	for _, p := range resp.Models {
		if p.Name == model {
			return true
		}
	}
	return false
}

// A projector named for the repo serves every tag beside it, so it goes only with the last of them.
func removeModel(m *store.Model) error {
	for _, p := range m.Parts {
		if err := os.Remove(p); err != nil {
			return err
		}
	}
	dir := filepath.Dir(m.ModelPath)
	if m.ProjectorPath != "" {
		keep, err := hasWeights(dir)
		if err != nil {
			return err
		}
		if !m.ProjectorShared || !keep {
			if err := os.Remove(m.ProjectorPath); err != nil && !errors.Is(err, os.ErrNotExist) {
				return err
			}
		}
	}
	// Only removed when empty. Anything else in there, an imatrix or a .part, was not this model's.
	if empty, _ := isEmptyDir(dir); empty {
		_ = os.Remove(dir)
	}
	return nil
}

func hasWeights(dir string) (bool, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return false, err
	}
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".gguf") {
			continue
		}
		n := store.ParseGGUFName(e.Name())
		if !n.Projector && !n.Imatrix && !n.Draft {
			return true, nil
		}
	}
	return false, nil
}

func isEmptyDir(dir string) (bool, error) {
	f, err := os.Open(dir)
	if err != nil {
		return false, err
	}
	defer f.Close()
	_, err = f.Readdirnames(1)
	return errors.Is(err, io.EOF), nil
}
