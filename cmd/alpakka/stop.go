package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"
)

const stopUsage = `alpakka stop [--force] [model]

Unloads the loaded model now rather than at its keep_alive. With no model,
unloads whatever is loaded; naming a model that is not loaded is an error.

A response still generating is let finish first, so stop can take as long as
that answer does. --force unloads at once and cuts the response off: its
client sees the stream end early.
`

func stop(args []string) error {
	fs := flag.NewFlagSet("stop", flag.ExitOnError)
	host := hostFlags(fs)
	force := fs.Bool("force", false, "unload at once, cutting off a response in progress")
	fs.Usage = func() {
		fmt.Fprint(fs.Output(), stopUsage)
		fs.PrintDefaults()
	}
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() > 1 {
		fs.Usage()
		return fmt.Errorf("stop takes at most one model")
	}
	addr, err := host()
	if err != nil {
		return err
	}
	model := fs.Arg(0)

	// Only a notice, so a failure to ask leaves it out rather than stopping the unload.
	var loaded psResponse
	if !*force && getJSON(addr, "/api/ps", &loaded) == nil {
		if name, busy := generating(loaded, model); busy {
			fmt.Fprintf(os.Stderr, "%s is generating: waiting for that response to finish before unloading.\n"+
				"Pass --force to cut it off, or Ctrl-C to leave it loaded.\n", name)
		}
	}

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()

	var resp struct {
		Model string `json:"model"`
	}
	req := map[string]any{"model": model, "force": *force}
	if err := postJSON(ctx, addr, "/alpakka/unload", req, &resp); err != nil {
		return err
	}
	if resp.Model == "" {
		fmt.Println("nothing was loaded")
		return nil
	}
	fmt.Println("unloaded", resp.Model)
	return nil
}

// The server matches a bare or partial name against the store, so this only needs to catch the
// exact spelling and the empty one. Any other name just goes without the notice.
func generating(resp psResponse, model string) (string, bool) {
	for _, m := range resp.Models {
		if m.Alpakka.Busy && (model == "" || model == m.Name || strings.HasPrefix(m.Name, model+":")) {
			return m.Name, true
		}
	}
	return "", false
}

type logsResponse struct {
	Model     string    `json:"model"`
	Running   bool      `json:"running"`
	StartedAt time.Time `json:"started_at"`
	Lines     []string  `json:"lines"`
}

func logs(args []string) error {
	fs := flag.NewFlagSet("logs", flag.ExitOnError)
	host := hostFlags(fs)
	n := fs.Int("n", 100, "number of lines, at most 400")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 0 {
		return fmt.Errorf("logs takes no arguments")
	}
	addr, err := host()
	if err != nil {
		return err
	}
	var resp logsResponse
	if err := getJSON(addr, "/alpakka/logs?n="+strconv.Itoa(*n), &resp); err != nil {
		return err
	}
	printLogs(os.Stdout, os.Stderr, resp, time.Now())
	return nil
}

// The header goes to stderr so the lines alone can be piped into grep.
func printLogs(out, meta io.Writer, resp logsResponse, now time.Time) {
	if resp.Model == "" {
		fmt.Fprintln(meta, "no llama-server has been started since alpakka did")
		return
	}
	state := "exited"
	if resp.Running {
		state = "running"
	}
	fmt.Fprintf(meta, "# %s, %s, started %s\n", resp.Model, state, ago(now.Sub(resp.StartedAt)))
	for _, line := range resp.Lines {
		fmt.Fprintln(out, line)
	}
}
