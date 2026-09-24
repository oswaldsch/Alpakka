package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"text/tabwriter"
	"time"

	"github.com/ollama/ollama/api"

	"github.com/oswaldsch/alpakka/internal/config"
)

// Both ask the running server rather than reading the store, since its -models flag
// can serve a different set of roots than the config names.
func list(args []string) error {
	host, err := serverHost("list", args)
	if err != nil {
		return err
	}
	var resp api.ListResponse
	if err := getJSON(host, "/api/tags", &resp); err != nil {
		return err
	}
	printList(os.Stdout, resp, time.Now())
	return nil
}

func ps(args []string) error {
	host, err := serverHost("ps", args)
	if err != nil {
		return err
	}
	var resp api.ProcessResponse
	if err := getJSON(host, "/api/ps", &resp); err != nil {
		return err
	}
	printPS(os.Stdout, resp, time.Now())
	return nil
}

func serverHost(name string, args []string) (string, error) {
	fs := flag.NewFlagSet(name, flag.ExitOnError)
	var (
		configPath = fs.String("config", config.DefaultPath(), "path to config.toml")
		host       = fs.String("host", "", "server address, default the configured listen address")
	)
	if err := fs.Parse(args); err != nil {
		return "", err
	}
	if fs.NArg() != 0 {
		return "", fmt.Errorf("%s takes no arguments", name)
	}
	if *host != "" {
		return *host, nil
	}
	cfg, err := config.Load(*configPath)
	if err != nil {
		return "", err
	}
	return cfg.Server.Listen, nil
}

func getJSON(host, path string, v any) error {
	client := http.Client{Timeout: 10 * time.Second}
	resp, err := client.Get("http://" + host + path)
	if err != nil {
		return fmt.Errorf("is alpakka running? %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return fmt.Errorf("%s: %s: %s", path, resp.Status, body)
	}
	return json.NewDecoder(resp.Body).Decode(v)
}

func printList(w io.Writer, resp api.ListResponse, now time.Time) {
	tw := tabwriter.NewWriter(w, 0, 0, 3, ' ', 0)
	fmt.Fprintln(tw, "NAME\tQUANT\tSIZE\tMODIFIED")
	for _, m := range resp.Models {
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\n", m.Name, m.Details.QuantizationLevel,
			humanBytes(m.Size), ago(now.Sub(m.ModifiedAt)))
	}
	tw.Flush()
}

func printPS(w io.Writer, resp api.ProcessResponse, now time.Time) {
	tw := tabwriter.NewWriter(w, 0, 0, 3, ' ', 0)
	fmt.Fprintln(tw, "NAME\tQUANT\tVRAM\tCONTEXT\tUNTIL")
	for _, m := range resp.Models {
		fmt.Fprintf(tw, "%s\t%s\t%s\t%d\t%s\n", m.Name, m.Details.QuantizationLevel,
			humanBytes(m.SizeVRAM), m.ContextLength, until(m.ExpiresAt, now))
	}
	tw.Flush()
}

func humanBytes(n int64) string {
	switch {
	case n >= 1e9:
		return fmt.Sprintf("%.1f GB", float64(n)/1e9)
	case n >= 1e6:
		return fmt.Sprintf("%.0f MB", float64(n)/1e6)
	default:
		return fmt.Sprintf("%d B", n)
	}
}

func ago(d time.Duration) string {
	switch {
	case d < time.Minute:
		return "just now"
	case d < time.Hour:
		return fmt.Sprintf("%d minutes ago", int(d.Minutes()))
	case d < 48*time.Hour:
		return fmt.Sprintf("%d hours ago", int(d.Hours()))
	default:
		return fmt.Sprintf("%d days ago", int(d.Hours()/24))
	}
}

func until(t, now time.Time) string {
	switch {
	case t.IsZero():
		return "forever"
	case !t.After(now):
		return "expired"
	default:
		return "in " + t.Sub(now).Round(time.Second).String()
	}
}
