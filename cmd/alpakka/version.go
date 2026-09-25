package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"regexp"
	"runtime/debug"
	"strings"
	"text/tabwriter"
	"time"

	alpakkaapi "github.com/oswaldsch/alpakka/internal/api"
	"github.com/oswaldsch/alpakka/internal/config"
)

func version(args []string) error {
	fs := flag.NewFlagSet("version", flag.ExitOnError)
	configPath := fs.String("config", config.DefaultPath(), "path to config.toml")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 0 {
		return fmt.Errorf("version takes no arguments")
	}

	info, ok := debug.ReadBuildInfo()
	fmt.Printf("%-14s%s\n", "alpakka", buildVersion(info, ok))
	fmt.Printf("%-14s%s\n", "ollama API", alpakkaapi.Version)

	cfg, err := config.Load(*configPath)
	if err != nil {
		return err
	}
	if cfg.Llama.LibDir == "" {
		return fmt.Errorf("%s: no lib_dir under [llama], so there is no llama-server to report on", *configPath)
	}
	fmt.Printf("%-14s%s\n", "llama-server", cfg.Llama.Binary())

	versionOut, err := llamaOutput(cfg.Llama, "--version")
	if err != nil {
		return fmt.Errorf("llama-server --version: %w\n%s", err, versionOut)
	}
	for _, line := range strings.Split(strings.TrimSpace(versionOut), "\n") {
		fmt.Printf("%-14s%s\n", "", line)
	}

	// Some builds exit non-zero after printing --help, so the text is what counts.
	help, _ := llamaOutput(cfg.Llama, "--help")
	if strings.TrimSpace(help) == "" {
		return fmt.Errorf("llama-server printed nothing for --help")
	}
	fmt.Println()
	return printFeatures(os.Stdout, llamaFeatures(help))
}

func buildVersion(info *debug.BuildInfo, ok bool) string {
	if !ok {
		return "unknown"
	}
	v := info.Main.Version
	var rev string
	var modified bool
	for _, s := range info.Settings {
		switch s.Key {
		case "vcs.revision":
			rev = s.Value
		case "vcs.modified":
			modified = s.Value == "true"
		}
	}
	if len(rev) > 12 {
		rev = rev[:12]
	}
	// Go stamps a pseudo-version that already names the commit and a dirty tree.
	if rev == "" || strings.Contains(v, rev) {
		return v
	}
	if modified {
		rev += ", modified"
	}
	return fmt.Sprintf("%s (%s)", v, rev)
}

// Run from the backend directory, as the supervisor does, so ggml finds the same libraries.
func llamaOutput(llama config.Llama, arg string) (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, llama.Binary(), arg)
	cmd.Dir = llama.BackendDir()
	out, err := cmd.CombinedOutput()
	return string(out), err
}

type feature struct {
	flag     string
	present  bool
	required bool
	// What goes missing without it.
	unlocks string
}

var flashAttnModes = regexp.MustCompile(`--flash-attn.*\b(on|off|auto)\b`)

// The same checks setup.sh makes: alpakka passes the required flags on every load, so a
// build without them rejects every request, while the rest only gate one option each.
func llamaFeatures(help string) []feature {
	has := func(flag string) bool { return strings.Contains(help, flag) }
	var out []feature
	for _, f := range []string{"--fit", "--spec-type", "--jinja", "--no-webui", "--no-mmproj"} {
		out = append(out, feature{flag: f, present: has(f), required: true})
	}
	out = append(out, feature{flag: "--flash-attn on|off|auto", present: flashAttnModes.MatchString(help), required: true})
	for _, f := range []feature{
		{flag: "--n-cpu-moe", unlocks: "num_cpu_moe"},
		{flag: "--override-tensor", unlocks: "override_tensor"},
		{flag: "--kv-stream-arena-mib", unlocks: "kv_stream_arena_mib"},
		{flag: "--moe-expert-cache", unlocks: "moe_expert_cache"},
	} {
		f.present = has(f.flag)
		out = append(out, f)
	}
	return out
}

func printFeatures(w io.Writer, features []feature) error {
	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	var missing []string
	for _, f := range features {
		state := "yes"
		switch {
		case f.present:
		case f.required:
			state = "MISSING"
			missing = append(missing, f.flag)
		default:
			state = "no, " + f.unlocks + " unavailable"
		}
		fmt.Fprintf(tw, "  %s\t%s\n", f.flag, state)
	}
	tw.Flush()
	if len(missing) > 0 {
		return fmt.Errorf("llama-server lacks %s, which alpakka passes on every load", strings.Join(missing, ", "))
	}
	return nil
}
