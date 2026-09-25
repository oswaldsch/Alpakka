package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"strconv"
	"syscall"
	"time"
)

func stop(args []string) error {
	fs := flag.NewFlagSet("stop", flag.ExitOnError)
	host := hostFlags(fs)
	fs.Usage = func() {
		fmt.Fprintln(fs.Output(), "alpakka stop [model]\n\nWith no model, unloads whatever is loaded.")
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

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()

	var resp struct {
		Model string `json:"model"`
	}
	if err := postJSON(ctx, addr, "/alpakka/unload", map[string]string{"model": fs.Arg(0)}, &resp); err != nil {
		return err
	}
	if resp.Model == "" {
		fmt.Println("nothing was loaded")
		return nil
	}
	fmt.Println("unloaded", resp.Model)
	return nil
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
