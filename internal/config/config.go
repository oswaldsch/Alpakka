// Package config holds alpakka's settings: where llama-server lives, and the
// per-model flag profiles that ollama has no way to express.
package config

import (
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/BurntSushi/toml"
)

// Config is the whole of ~/.config/alpakka/config.toml.
type Config struct {
	Server   Server             `toml:"server"`
	Llama    Llama              `toml:"llama"`
	Defaults Profile            `toml:"defaults"`
	Models   map[string]Profile `toml:"models"`
}

// Server is where alpakka listens.
type Server struct {
	Listen string `toml:"listen"`
}

// Llama locates the llama.cpp build to drive.
type Llama struct {
	// LibDir holds llama-server and the per-backend subdirectories.
	LibDir string `toml:"lib_dir"`
	// Backend names the subdirectory of LibDir holding the ggml backend
	// libraries, e.g. "rocm_v7_2" or "vulkan".
	Backend string `toml:"backend"`
}

// Binary is the llama-server executable.
func (l Llama) Binary() string { return filepath.Join(l.LibDir, "llama-server") }

// BackendDir is the directory the server must run from. ggml looks for
// libggml-hip.so in the executable's directory and the working directory; if it
// finds neither the server starts on CPU at a few tokens a second and says
// nothing about it.
func (l Llama) BackendDir() string { return filepath.Join(l.LibDir, l.Backend) }

// Profile is a set of overrides. Every field is a pointer so that "unset" is
// distinguishable from "set to the zero value", which is what lets a request's
// options layer cleanly over a model's defaults over the global defaults.
type Profile struct {
	// Process-level: changing any of these requires a llama-server restart.
	NumCtx        *int    `toml:"num_ctx"`
	CacheTypeK    *string `toml:"cache_type_k"`
	CacheTypeV    *string `toml:"cache_type_v"`
	SpecType      *string `toml:"spec_type"`
	SpecDraftNMax *int    `toml:"spec_draft_n_max"`
	SpecDraftNMin *int    `toml:"spec_draft_n_min"`
	NumGPU        *int    `toml:"num_gpu"`
	Parallel      *int    `toml:"parallel"`
	FlashAttn     *string `toml:"flash_attn"`
	Projector     *bool   `toml:"projector"`
	Backend       *string `toml:"backend"`

	// Request-level: applied per request, no restart.
	ReasoningEffort *string  `toml:"reasoning_effort"`
	Temperature     *float32 `toml:"temperature"`
	TopK            *int     `toml:"top_k"`
	TopP            *float32 `toml:"top_p"`
	MinP            *float32 `toml:"min_p"`
	RepeatPenalty   *float32 `toml:"repeat_penalty"`
	Seed            *int     `toml:"seed"`
	NumPredict      *int     `toml:"num_predict"`
	Stop            []string `toml:"stop"`

	KeepAlive *string `toml:"keep_alive"`
}

// Default is the configuration used when no file exists. It encodes the machine
// this was built for: ollama's llama.cpp build on the ROCm backend, and the
// settings the benchmarks in gfx1200-lab showed to be worth having.
func Default() Config {
	return Config{
		Server: Server{Listen: "127.0.0.1:11435"},
		Llama:  Llama{LibDir: "/usr/local/lib/ollama", Backend: "rocm_v7_2"},
		Defaults: Profile{
			NumCtx:     ptr(32768),
			CacheTypeK: ptr("q8_0"),
			CacheTypeV: ptr("q8_0"),
			NumGPU:     ptr(99),
			Parallel:   ptr(1),
			FlashAttn:  ptr("on"),
			KeepAlive:  ptr("5m"),
		},
		Models: map[string]Profile{},
	}
}

// DefaultPath is the config file location, honouring XDG_CONFIG_HOME.
func DefaultPath() string {
	dir := os.Getenv("XDG_CONFIG_HOME")
	if dir == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return "alpakka.toml"
		}
		dir = filepath.Join(home, ".config")
	}
	return filepath.Join(dir, "alpakka", "config.toml")
}

// Load reads the config at path, layering it over Default. A missing file is
// not an error: alpakka runs on its defaults.
func Load(path string) (Config, error) {
	cfg := Default()

	b, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return cfg, nil
	}
	if err != nil {
		return cfg, err
	}

	var file Config
	if _, err := toml.Decode(string(b), &file); err != nil {
		return cfg, fmt.Errorf("%s: %w", path, err)
	}

	if file.Server.Listen != "" {
		cfg.Server.Listen = file.Server.Listen
	}
	if file.Llama.LibDir != "" {
		cfg.Llama.LibDir = file.Llama.LibDir
	}
	if file.Llama.Backend != "" {
		cfg.Llama.Backend = file.Llama.Backend
	}
	cfg.Defaults = Merge(cfg.Defaults, file.Defaults)
	for name, p := range file.Models {
		cfg.Models[name] = p
	}

	if err := cfg.Validate(); err != nil {
		return cfg, fmt.Errorf("%s: %w", path, err)
	}
	return cfg, nil
}

// ForModel returns the global defaults with the model's own overrides applied.
func (c Config) ForModel(name string) Profile {
	p := c.Defaults
	if over, ok := c.Models[name]; ok {
		p = Merge(p, over)
	}
	// A config may key a model by its bare name while requests use name:tag.
	if bare, _, ok := splitTag(name); ok {
		if over, ok := c.Models[bare]; ok {
			p = Merge(p, over)
		}
	}
	return p
}

func splitTag(name string) (string, string, bool) {
	for i := len(name) - 1; i >= 0; i-- {
		switch name[i] {
		case ':':
			return name[:i], name[i+1:], true
		case '/':
			return "", "", false
		}
	}
	return "", "", false
}

// Merge layers over onto base; only fields set in over take effect.
func Merge(base, over Profile) Profile {
	out := base
	setIf(&out.NumCtx, over.NumCtx)
	setIf(&out.CacheTypeK, over.CacheTypeK)
	setIf(&out.CacheTypeV, over.CacheTypeV)
	setIf(&out.SpecType, over.SpecType)
	setIf(&out.SpecDraftNMax, over.SpecDraftNMax)
	setIf(&out.SpecDraftNMin, over.SpecDraftNMin)
	setIf(&out.NumGPU, over.NumGPU)
	setIf(&out.Parallel, over.Parallel)
	setIf(&out.FlashAttn, over.FlashAttn)
	setIf(&out.Projector, over.Projector)
	setIf(&out.Backend, over.Backend)
	setIf(&out.ReasoningEffort, over.ReasoningEffort)
	setIf(&out.Temperature, over.Temperature)
	setIf(&out.TopK, over.TopK)
	setIf(&out.TopP, over.TopP)
	setIf(&out.MinP, over.MinP)
	setIf(&out.RepeatPenalty, over.RepeatPenalty)
	setIf(&out.Seed, over.Seed)
	setIf(&out.NumPredict, over.NumPredict)
	setIf(&out.KeepAlive, over.KeepAlive)
	if over.Stop != nil {
		out.Stop = over.Stop
	}
	return out
}

func setIf[T any](dst **T, src *T) {
	if src != nil {
		*dst = src
	}
}

func ptr[T any](v T) *T { return &v }

// ValidEfforts are the only values the Qwen3.8 template accepts. It rejects
// anything else outright and silently promotes "high" to "xhigh", so alpakka
// refuses the values that would not mean what the caller intended.
var ValidEfforts = []string{"low", "medium", "xhigh"}

// Validate checks the values alpakka can reject before a slow model load.
func (c Config) Validate() error {
	if err := c.Defaults.Validate(); err != nil {
		return fmt.Errorf("defaults: %w", err)
	}
	for name, p := range c.Models {
		if err := p.Validate(); err != nil {
			return fmt.Errorf("models.%q: %w", name, err)
		}
	}
	return nil
}

// Validate checks a single profile.
func (p Profile) Validate() error {
	if p.ReasoningEffort != nil {
		if !valid(*p.ReasoningEffort, ValidEfforts) {
			return fmt.Errorf("reasoning_effort %q: must be one of %v",
				*p.ReasoningEffort, ValidEfforts)
		}
	}
	if p.NumCtx != nil && *p.NumCtx <= 0 {
		return fmt.Errorf("num_ctx must be positive, got %d", *p.NumCtx)
	}
	if p.KeepAlive != nil {
		if _, err := time.ParseDuration(*p.KeepAlive); err != nil {
			return fmt.Errorf("keep_alive %q: %w", *p.KeepAlive, err)
		}
	}
	return nil
}

func valid(v string, allowed []string) bool {
	for _, a := range allowed {
		if v == a {
			return true
		}
	}
	return false
}

// KeepAliveDuration resolves the profile's keep-alive, falling back to ollama's
// own five-minute default.
func (p Profile) KeepAliveDuration() time.Duration {
	if p.KeepAlive == nil {
		return 5 * time.Minute
	}
	d, err := time.ParseDuration(*p.KeepAlive)
	if err != nil {
		return 5 * time.Minute
	}
	return d
}

// Runtime is the fully resolved set of process-level settings for one
// llama-server. It is deliberately comparable: the supervisor decides whether a
// request can be served by the running process by comparing two of these, so a
// changed flag causes a clean reload instead of being silently ignored until
// the next cold start.
type Runtime struct {
	Model         string // canonical model name, for logs and /api/ps
	ModelPath     string
	ProjectorPath string // empty when the projector is disabled or absent
	NumCtx        int
	CacheTypeK    string
	CacheTypeV    string
	SpecType      string
	SpecDraftNMax int
	SpecDraftNMin int
	NumGPU        int
	Parallel      int
	FlashAttn     string
	Backend       string
}

// Runtime resolves the profile's process-level settings for a model.
// projectorPath is the projector the store found; the profile can suppress it,
// which on a 16 GB card is the difference between fitting and not.
func (p Profile) Runtime(model, modelPath, projectorPath string) Runtime {
	rt := Runtime{
		Model:         model,
		ModelPath:     modelPath,
		ProjectorPath: projectorPath,
		NumCtx:        deref(p.NumCtx, 32768),
		CacheTypeK:    deref(p.CacheTypeK, "q8_0"),
		CacheTypeV:    deref(p.CacheTypeV, "q8_0"),
		SpecType:      deref(p.SpecType, ""),
		SpecDraftNMax: deref(p.SpecDraftNMax, 0),
		SpecDraftNMin: deref(p.SpecDraftNMin, 0),
		NumGPU:        deref(p.NumGPU, 99),
		Parallel:      deref(p.Parallel, 1),
		FlashAttn:     deref(p.FlashAttn, "on"),
		Backend:       deref(p.Backend, ""),
	}
	if p.Projector != nil && !*p.Projector {
		rt.ProjectorPath = ""
	}
	return rt
}

func deref[T any](p *T, fallback T) T {
	if p == nil {
		return fallback
	}
	return *p
}
