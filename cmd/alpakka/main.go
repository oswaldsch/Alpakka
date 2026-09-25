// Command alpakka serves ollama's API backed by llama-server.
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
	"slices"
	"strings"
	"syscall"
	"text/tabwriter"
	"time"

	"github.com/oswaldsch/alpakka/internal/api"
	"github.com/oswaldsch/alpakka/internal/config"
	"github.com/oswaldsch/alpakka/internal/store"
	"github.com/oswaldsch/alpakka/internal/supervisor"
)

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, "alpakka:", err)
		os.Exit(1)
	}
}

type command struct {
	names   []string
	summary string
	run     func([]string) error
}

// A function rather than a variable, since help lists the table it is in.
func commands() []command {
	return []command{
		{[]string{"serve"}, "run the server (the default with no command)", serve},
		{[]string{"pull"}, "fetch a GGUF from HuggingFace into the first model root", pull},
		{[]string{"list", "ls"}, "list the models the server can serve", list},
		{[]string{"ps"}, "show the loaded model", ps},
		{[]string{"show"}, "show a model's details, template or Modelfile", show},
		{[]string{"run"}, "chat with a model, or answer one prompt", runModel},
		{[]string{"stop"}, "unload the loaded model now", stop},
		{[]string{"rm"}, "delete a model from its root", rm},
		{[]string{"logs"}, "print llama-server's log for the last load", logs},
		{[]string{"version"}, "print alpakka's and llama-server's versions", version},
		{[]string{"help"}, "list the commands", help},
	}
}

func run(args []string) error {
	// Bare flags still start the server, as existing unit files invoke it.
	if len(args) == 0 || strings.HasPrefix(args[0], "-") {
		return serve(args)
	}
	name, rest := args[0], args[1:]
	for _, c := range commands() {
		if slices.Contains(c.names, name) {
			return c.run(rest)
		}
	}
	return fmt.Errorf("unknown command %q: run `alpakka help` for the list", name)
}

func help([]string) error {
	fmt.Println("usage: alpakka <command> [flags] [args]")
	fmt.Println()
	tw := tabwriter.NewWriter(os.Stdout, 0, 0, 3, ' ', 0)
	for _, c := range commands() {
		fmt.Fprintf(tw, "  %s\t%s\n", strings.Join(c.names, ", "), c.summary)
	}
	tw.Flush()
	fmt.Println()
	fmt.Println("`alpakka <command> -h` lists a command's flags.")
	return nil
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

	if cfg.Llama.LibDir == "" {
		return fmt.Errorf("%s: set lib_dir under [llama] to the directory holding llama-server", *configPath)
	}
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
		Store:  store.New(logger.Printf, roots...),
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
		// No read or write deadline, since generation can run for minutes.
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

	// The child holds the whole card, so leaving it running breaks the next start.
	sup.Stop()
	return nil
}

// A configured root that does not exist yet is skipped, not fatal.
func openRoots(cfg config.Config, override string, logger *log.Logger) ([]string, error) {
	roots := cfg.Store.Roots
	if override != "" {
		roots = []string{override}
	}
	if len(roots) == 0 {
		roots = store.DefaultRoots()
	}

	var readable []string
	for _, root := range roots {
		if _, err := os.Stat(root); err != nil {
			logger.Printf("model root %s: skipped (%v)", root, err)
			continue
		}
		logger.Printf("model root:   %s", root)
		readable = append(readable, root)
	}
	if len(readable) == 0 {
		return nil, fmt.Errorf("no readable model root among %v", roots)
	}
	return readable, nil
}
