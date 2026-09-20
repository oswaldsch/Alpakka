package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"

	"github.com/oswald/alpakka/internal/config"
	"github.com/oswald/alpakka/internal/store"
)

func importModels(args []string) error {
	fs := flag.NewFlagSet("import", flag.ExitOnError)
	var (
		configPath = fs.String("config", config.DefaultPath(), "path to config.toml")
		fromOllama = fs.Bool("from-ollama", false, "import from ollama's model store")
		from       = fs.String("from", "", "ollama root to import from")
		root       = fs.String("root", "", "write into this root instead of the configured one")
		apply      = fs.Bool("apply", false, "create the links instead of only listing them")
	)
	if err := fs.Parse(args); err != nil {
		return err
	}
	if !*fromOllama && *from == "" {
		return fmt.Errorf("import needs --from-ollama")
	}

	cfg, err := config.Load(*configPath)
	if err != nil {
		return err
	}
	src := *from
	if src == "" {
		src = ollamaRoot(cfg)
	}
	dest, err := writableRoot(cfg, *root)
	if err != nil {
		return err
	}
	if src == dest {
		return fmt.Errorf("%s is both the source and the destination", src)
	}

	logger := log.New(os.Stderr, "", log.LstdFlags)
	actions, err := store.PlanImport(store.New(src), dest)
	if err != nil {
		return err
	}

	var linked, skipped, failed int
	for _, a := range actions {
		if a.Skip != "" {
			logger.Printf("skip %s: %s", a.Model, a.Skip)
			skipped++
			continue
		}
		for _, l := range a.Links {
			logger.Printf("link %s -> %s", l.From, l.To)
		}
		if *apply {
			// One model that cannot be linked should not cost the rest.
			if err := a.Link(); err != nil {
				logger.Printf("failed %v", err)
				failed++
				continue
			}
		}
		linked++
	}

	verb := "would link"
	if *apply {
		verb = "linked"
	}
	logger.Printf("%s %d models into %s, skipped %d, failed %d", verb, linked, dest, skipped, failed)
	if !*apply && linked > 0 {
		logger.Printf("pass --apply to create the links")
	}
	if failed > 0 {
		return fmt.Errorf("%d of %d models could not be linked", failed, failed+linked)
	}
	return nil
}

// ollamaRoot is the first configured root in ollama's layout, falling back to
// where ollama keeps it by default.
func ollamaRoot(cfg config.Config) string {
	for _, root := range cfg.Store.Roots {
		if info, err := os.Stat(filepath.Join(root, "manifests")); err == nil && info.IsDir() {
			return root
		}
	}
	return store.DefaultRoot()
}
