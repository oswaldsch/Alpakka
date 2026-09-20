// Command alpakka serves ollama's API backed by llama-server, with the
// performance settings ollama has no way to express.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/oswald/alpakka/internal/api"
	"github.com/oswald/alpakka/internal/config"
	"github.com/oswald/alpakka/internal/store"
	"github.com/oswald/alpakka/internal/supervisor"
)

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, "alpakka:", err)
		os.Exit(1)
	}
}

func run(args []string) error {
	// Bare flags keep starting the server, which is what every existing unit
	// file and script invokes.
	if len(args) == 0 || strings.HasPrefix(args[0], "-") {
		return serve(args)
	}
	cmd, rest := args[0], args[1:]
	switch cmd {
	case "serve":
		return serve(rest)
	case "pull":
		return pull(rest)
	case "import":
		return importModels(rest)
	}
	return fmt.Errorf("unknown command %q: expected serve, pull or import", cmd)
}

func serve(args []string) error {
	fs := flag.NewFlagSet("serve", flag.ExitOnError)
	var (
		configPath = fs.String("config", config.DefaultPath(), "path to config.toml")
		listen     = fs.String("listen", "", "override the configured listen address")
		modelsRoot = fs.String("models", "", "read this model root instead of the configured ones")
	)
	if err := fs.Parse(args); err != nil {
		return err
	}

	cfg, err := config.Load(*configPath)
	if err != nil {
		return err
	}
	if *listen != "" {
		cfg.Server.Listen = *listen
	}

	logger := log.New(os.Stderr, "", log.LstdFlags)

	if _, err := os.Stat(cfg.Llama.Binary()); err != nil {
		return fmt.Errorf("llama-server not found at %s: %w", cfg.Llama.Binary(), err)
	}
	if _, err := os.Stat(cfg.Llama.BackendDir()); err != nil {
		return fmt.Errorf("backend directory %s not found: %w", cfg.Llama.BackendDir(), err)
	}

	roots, err := openRoots(cfg, *modelsRoot, logger)
	if err != nil {
		return err
	}

	sup := supervisor.New(cfg.Llama, cfg.WoL, logger.Printf)
	srv := &api.Server{
		Store:  store.NewMulti(logger.Printf, roots...),
		Config: cfg,
		Super:  sup,
		Logger: logger,
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	go sup.RunEvictor(ctx, 10*time.Second)

	httpServer := &http.Server{
		Addr:    cfg.Server.Listen,
		Handler: srv.Handler(),
		// Generation can legitimately run for minutes, so neither the read nor
		// the write side may impose a deadline.
		ReadHeaderTimeout: 15 * time.Second,
	}

	errc := make(chan error, 1)
	go func() {
		logger.Printf("alpakka listening on %s", cfg.Server.Listen)
		logger.Printf("llama-server: %s (backend %s)", cfg.Llama.Binary(), cfg.Llama.Backend)
		if err := httpServer.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			errc <- err
		}
	}()

	select {
	case err := <-errc:
		return err
	case <-ctx.Done():
		logger.Printf("shutting down")
	}

	shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	_ = httpServer.Shutdown(shutdownCtx)

	// The child holds the whole card. Leaving it running would make the next
	// start fail for reasons that look nothing like the cause.
	sup.Stop()
	return nil
}

// openRoots resolves the roots to serve and reports which layout each is in.
// A configured root that does not exist yet is skipped rather than fatal: a
// machine may have ~/models before anything has been pulled into it, or the
// other way round.
func openRoots(cfg config.Config, override string, logger *log.Logger) ([]store.Source, error) {
	roots := cfg.Store.Roots
	if override != "" {
		roots = []string{override}
	}
	if len(roots) == 0 {
		roots = store.DefaultRoots()
	}

	var sources []store.Source
	for _, root := range roots {
		if _, err := os.Stat(root); err != nil {
			logger.Printf("model root %s: skipped (%v)", root, err)
			continue
		}
		src := store.Open(root)
		kind := "gguf directory"
		if _, ok := src.(*store.Store); ok {
			kind = "ollama store"
		}
		logger.Printf("model root:   %s (%s)", root, kind)
		sources = append(sources, src)
	}
	if len(sources) == 0 {
		return nil, fmt.Errorf("no readable model root among %v", roots)
	}
	return sources, nil
}
