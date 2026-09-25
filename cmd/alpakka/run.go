package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"os/signal"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
)

const runUsage = `alpakka run [flags] <model> [prompt...]

With a prompt, or one piped on stdin, answers it and exits; given both, the
piped text follows the prompt. Otherwise starts a chat: /clear forgets the conversation, /bye or Ctrl-D leaves, and Ctrl-C stops
an answer without leaving.

Flags go before the model.
`

func runModel(args []string) error {
	fs := flag.NewFlagSet("run", flag.ExitOnError)
	host := hostFlags(fs)
	opts := optionFlags{}
	fs.Var(opts, "o", "an option as key=value, repeatable: -o reasoning_effort=low -o num_ctx=65536")
	var (
		system       = fs.String("system", "", "system prompt")
		keepAlive    = fs.String("keepalive", "", "how long the model stays loaded afterwards, e.g. 30m, or 0 to unload")
		verbose      = fs.Bool("verbose", false, "print token counts and rates after each answer")
		hideThinking = fs.Bool("hide-thinking", false, "leave the model's reasoning out")
	)
	fs.Usage = func() {
		fmt.Fprint(fs.Output(), runUsage)
		fs.PrintDefaults()
	}
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 1 {
		fs.Usage()
		return fmt.Errorf("run needs a model")
	}
	addr, err := host()
	if err != nil {
		return err
	}

	c := &chatSession{
		client:  api.NewClient(&url.URL{Scheme: "http", Host: addr}, http.DefaultClient),
		model:   fs.Arg(0),
		options: opts,
		out:     os.Stdout,
		meta:    os.Stderr,
		verbose: *verbose,
	}
	if !*hideThinking {
		// Kept off stdout, so a piped answer is only the answer.
		c.thinking = os.Stderr
	}
	if *keepAlive != "" {
		d, err := time.ParseDuration(*keepAlive)
		if err != nil {
			return fmt.Errorf("-keepalive: %w", err)
		}
		c.keepAlive = &api.Duration{Duration: d}
	}
	if *system != "" {
		c.history = []api.Message{{Role: "system", Content: *system}}
	}

	prompt := strings.Join(fs.Args()[1:], " ")
	if !onTerminal() {
		in, err := io.ReadAll(os.Stdin)
		if err != nil {
			return err
		}
		prompt = joinPrompt(prompt, string(in))
		if prompt == "" {
			return fmt.Errorf("no prompt given and nothing on stdin")
		}
	}
	if prompt != "" {
		ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
		defer stop()
		return c.turn(ctx, prompt)
	}
	return c.repl(os.Stdin)
}

// Piped input follows the instruction, so `git diff | alpakka run m "review this"` reads naturally.
func joinPrompt(arg, stdin string) string {
	stdin = strings.TrimSpace(stdin)
	switch {
	case arg == "":
		return stdin
	case stdin == "":
		return arg
	default:
		return arg + "\n\n" + stdin
	}
}

type chatSession struct {
	client    *api.Client
	model     string
	options   map[string]any
	keepAlive *api.Duration
	history   []api.Message

	out      io.Writer
	thinking io.Writer // nil hides it
	meta     io.Writer
	verbose  bool
}

func (c *chatSession) repl(in io.Reader) error {
	scanner := bufio.NewScanner(in)
	// A pasted file is one line as far as this is concerned.
	scanner.Buffer(make([]byte, 64*1024), 16*1024*1024)
	for {
		fmt.Fprint(c.meta, ">>> ")
		if !scanner.Scan() {
			fmt.Fprintln(c.meta)
			return scanner.Err()
		}
		line := strings.TrimSpace(scanner.Text())
		switch line {
		case "":
			continue
		case "/bye", "/exit":
			return nil
		case "/clear":
			c.history = systemOnly(c.history)
			fmt.Fprintln(c.meta, "cleared the conversation")
			continue
		case "/?", "/help":
			fmt.Fprintln(c.meta, "/clear forgets the conversation, /bye leaves")
			continue
		}

		// Ctrl-C ends the answer rather than the session.
		ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
		err := c.turn(ctx, line)
		stop()
		switch {
		case errors.Is(err, context.Canceled):
			fmt.Fprintln(c.meta, "\n(stopped)")
		case err != nil:
			fmt.Fprintln(c.meta, "error:", err)
		}
	}
}

func systemOnly(history []api.Message) []api.Message {
	if len(history) > 0 && history[0].Role == "system" {
		return history[:1]
	}
	return nil
}

// A turn that fails leaves the history as it was, so the question can be asked again.
func (c *chatSession) turn(ctx context.Context, prompt string) error {
	messages := append(c.history[:len(c.history):len(c.history)], api.Message{Role: "user", Content: prompt})
	stream := true
	req := &api.ChatRequest{
		Model:     c.model,
		Messages:  messages,
		Stream:    &stream,
		Options:   c.options,
		KeepAlive: c.keepAlive,
	}

	var answer strings.Builder
	var final api.ChatResponse
	thought := false
	err := c.client.Chat(ctx, req, func(r api.ChatResponse) error {
		if r.Message.Thinking != "" && c.thinking != nil {
			fmt.Fprint(c.thinking, r.Message.Thinking)
			thought = true
		}
		if r.Message.Content != "" {
			if thought {
				fmt.Fprint(c.thinking, "\n\n")
				thought = false
			}
			fmt.Fprint(c.out, r.Message.Content)
			answer.WriteString(r.Message.Content)
		}
		if r.Done {
			final = r
		}
		return nil
	})
	if err != nil {
		return err
	}
	fmt.Fprintln(c.out)
	if c.verbose {
		fmt.Fprintln(c.meta, stats(final.Metrics))
	}
	c.history = append(messages, api.Message{Role: "assistant", Content: answer.String()})
	return nil
}

func stats(m api.Metrics) string {
	s := fmt.Sprintf("prompt %d tokens, %s · decode %d tokens, %s",
		m.PromptEvalCount, perSecond(m.PromptEvalCount, m.PromptEvalDuration),
		m.EvalCount, perSecond(m.EvalCount, m.EvalDuration))
	if m.LoadDuration > 0 {
		s += " · load " + m.LoadDuration.Round(time.Millisecond).String()
	}
	return s
}

func perSecond(n int, d time.Duration) string {
	if d <= 0 {
		return "- tokens/s"
	}
	return fmt.Sprintf("%.2f tokens/s", float64(n)/d.Seconds())
}

// Values are read as JSON where they parse, so numbers, booleans and lists arrive typed,
// and as plain strings otherwise, so -o reasoning_effort=low needs no quoting.
type optionFlags map[string]any

func (o optionFlags) String() string {
	b, _ := json.Marshal(map[string]any(o))
	return string(b)
}

func (o optionFlags) Set(s string) error {
	k, v, ok := strings.Cut(s, "=")
	if !ok || k == "" {
		return fmt.Errorf("%q: want key=value", s)
	}
	var parsed any
	if err := json.Unmarshal([]byte(v), &parsed); err != nil {
		parsed = v
	}
	o[k] = parsed
	return nil
}
