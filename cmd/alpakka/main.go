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
	"syscall"
	"time"

	"github.com/oswald/alpakka/internal/api"
	"github.com/oswald/alpakka/internal/config"
	"github.com/oswald/alpakka/internal/store"
	"github.com/oswald/alpakka/internal/supervisor"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "alpakka:", err)
		os.Exit(1)
	}
}

func run() error {
	var (
		configPath = flag.String("config", config.DefaultPath(), "path to config.toml")
		listen     = flag.String("listen", "", "override the configured listen address")
		modelsRoot = flag.String("models", store.DefaultRoot(), "ollama model store to read")
	)
	flag.Parse()

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
	if _, err := os.Stat(*modelsRoot); err != nil {
		return fmt.Errorf("model store %s not readable: %w", *modelsRoot, err)
	}

	sup := supervisor.New(cfg.Llama, logger.Printf)
	srv := &api.Server{
		Store:  store.New(*modelsRoot),
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
		logger.Printf("model store:  %s", *modelsRoot)
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
