package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"

	"github.com/oswald/alpakka/internal/config"
	"github.com/oswald/alpakka/internal/hub"
	"github.com/oswald/alpakka/internal/store"
)

const pullUsage = `alpakka pull <ref>

  https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B-UD-IQ3_XXS.gguf
  https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-UD-IQ3_XXS.gguf
  hf.co/unsloth/Qwen3.8-27B-GGUF/UD-IQ3_XXS
  unsloth/Qwen3.8-27B-GGUF:UD-IQ3_XXS
`

func pull(args []string) error {
	fs := flag.NewFlagSet("pull", flag.ExitOnError)
	var (
		configPath = fs.String("config", config.DefaultPath(), "path to config.toml")
		root       = fs.String("root", "", "write into this root instead of the configured one")
		force      = fs.Bool("f", false, "replace an existing tag")
		mmproj     = fs.String("mmproj", "auto", "pull the repo's projector: yes, no or auto")
	)
	fs.Usage = func() {
		fmt.Fprint(fs.Output(), pullUsage)
		fs.PrintDefaults()
	}
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 1 {
		fs.Usage()
		return fmt.Errorf("pull takes exactly one model reference")
	}

	ref, err := hub.ParseRef(fs.Arg(0))
	if err != nil {
		return err
	}
	cfg, err := config.Load(*configPath)
	if err != nil {
		return err
	}
	dest, err := writableRoot(cfg, *root)
	if err != nil {
		return err
	}

	logger := log.New(os.Stderr, "", log.LstdFlags)
	client := hub.NewClient(logger.Printf)

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	plan, err := client.Resolve(ctx, ref)
	if err != nil {
		return err
	}

	withProjector := false
	if plan.Projector != nil {
		withProjector, err = wantProjector(*mmproj, plan, logger)
		if err != nil {
			return err
		}
	}
	if err := os.MkdirAll(dest, 0o755); err != nil {
		return err
	}
	return client.Pull(ctx, plan, dest, withProjector, *force)
}

// wantProjector decides whether to take the repo's mmproj beside the weights.
// Without a terminal to ask, it is left out: a projector only matters for a
// vision model, and an unattended pull should not double its own size.
func wantProjector(mode string, plan *hub.Plan, logger *log.Logger) (bool, error) {
	switch strings.ToLower(mode) {
	case "yes", "true":
		return true, nil
	case "no", "false":
		return false, nil
	case "auto":
	default:
		return false, fmt.Errorf("-mmproj %q: expected yes, no or auto", mode)
	}

	if !onTerminal() {
		logger.Printf("pull %s: repo has a projector (%s), pass -mmproj=yes to take it",
			plan.Repo, plan.Projector.Path)
		return false, nil
	}
	fmt.Fprintf(os.Stderr, "%s has a vision projector (%s). Pull it too? [y/N] ",
		plan.Repo, plan.Projector.Path)
	answer, err := bufio.NewReader(os.Stdin).ReadString('\n')
	if err != nil {
		return false, nil
	}
	answer = strings.ToLower(strings.TrimSpace(answer))
	return answer == "y" || answer == "yes", nil
}

func onTerminal() bool {
	info, err := os.Stdin.Stat()
	return err == nil && info.Mode()&os.ModeCharDevice != 0
}

// writableRoot is the first root that is not an ollama store, which is where
// a pull lands. Ollama's layout is read-only to alpakka, so it is never a
// candidate however it is ordered.
func writableRoot(cfg config.Config, override string) (string, error) {
	if override != "" {
		return override, nil
	}
	roots := cfg.Store.Roots
	if len(roots) == 0 {
		roots = store.DefaultRoots()
	}
	for _, root := range roots {
		if info, err := os.Stat(filepath.Join(root, "manifests")); err == nil && info.IsDir() {
			continue
		}
		return root, nil
	}
	return "", fmt.Errorf("no directory-backed model root among %v", roots)
}
