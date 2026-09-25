package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"slices"
	"text/tabwriter"
	"time"

	"github.com/ollama/ollama/api"
)

func show(args []string) error {
	fs := flag.NewFlagSet("show", flag.ExitOnError)
	host := hostFlags(fs)
	var (
		template  = fs.Bool("template", false, "print the chat template from the GGUF")
		modelfile = fs.Bool("modelfile", false, "print the generated Modelfile")
		info      = fs.Bool("info", false, "print every GGUF metadata key")
	)
	fs.Usage = func() {
		fmt.Fprintln(fs.Output(), "alpakka show [-template | -modelfile | -info] <model>")
		fs.PrintDefaults()
	}
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 1 {
		fs.Usage()
		return fmt.Errorf("show takes exactly one model")
	}
	addr, err := host()
	if err != nil {
		return err
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	var resp api.ShowResponse
	if err := postJSON(ctx, addr, "/api/show", api.ShowRequest{Model: fs.Arg(0)}, &resp); err != nil {
		return err
	}

	switch {
	case *template:
		fmt.Println(resp.Template)
	case *modelfile:
		fmt.Print(resp.Modelfile)
	case *info:
		printModelInfo(os.Stdout, resp.ModelInfo)
	default:
		printShow(os.Stdout, fs.Arg(0), resp)
	}
	return nil
}

func printShow(w io.Writer, name string, resp api.ShowResponse) {
	d := resp.Details
	tw := tabwriter.NewWriter(w, 0, 0, 3, ' ', 0)
	fmt.Fprintln(tw, "  Model")
	row := func(k string, v any) {
		if v != "" && v != 0 {
			fmt.Fprintf(tw, "    %s\t%v\n", k, v)
		}
	}
	row("name", name)
	row("architecture", d.Family)
	row("parameters", d.ParameterSize)
	row("quantization", d.QuantizationLevel)
	row("context length", d.ContextLength)
	row("embedding length", d.EmbeddingLength)
	if n, ok := resp.ModelInfo[d.Family+".block_count"].(float64); ok {
		row("layers", int(n))
	}
	if !resp.ModifiedAt.IsZero() {
		row("modified", resp.ModifiedAt.Local().Format(time.DateTime))
	}

	if len(resp.Capabilities) > 0 {
		fmt.Fprintln(tw, "\n  Capabilities")
		for _, c := range resp.Capabilities {
			fmt.Fprintf(tw, "    %s\n", c)
		}
	}
	tw.Flush()
}

func printModelInfo(w io.Writer, info map[string]any) {
	keys := make([]string, 0, len(info))
	for k := range info {
		keys = append(keys, k)
	}
	slices.Sort(keys)
	tw := tabwriter.NewWriter(w, 0, 0, 3, ' ', 0)
	for _, k := range keys {
		v := []rune(fmt.Sprint(info[k]))
		// Vocabularies are tens of thousands of entries long.
		if len(v) > 120 {
			v = append(v[:117], []rune("...")...)
		}
		fmt.Fprintf(tw, "%s\t%s\n", k, string(v))
	}
	tw.Flush()
}
