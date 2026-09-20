// Package config holds alpakka's settings: where llama-server lives, and the
// per-model flag profiles that ollama has no way to express.
package config

import (
	"fmt"
	"net"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"github.com/BurntSushi/toml"
)

// Config is the whole of ~/.config/alpakka/config.toml.
type Config struct {
	Server Server `toml:"server"`
	Store  Store  `toml:"store"`
	Llama  Llama  `toml:"llama"`
	// WoL maps an RPC endpoint's "host:port", as it appears in a profile's
	// rpc_servers, to the MAC address of the machine behind it. A load that
	// offloads onto the endpoint sends that address a Wake-on-LAN magic packet
	// first, on the LAN broadcast address, so a sleeping node does not have to
	// be woken by hand.
	WoL      map[string]string  `toml:"wol"`
	Defaults Profile            `toml:"defaults"`
	Models   map[string]Profile `toml:"models"`
}

// Server is where alpakka listens.
type Server struct {
	Listen string `toml:"listen"`
	// Origins are the browser origins allowed to call the API. Empty means
	// ollama's own default set: any port on localhost, plus the schemes
	// desktop clients use. Browser clients cannot talk to alpakka at all
	// without this, and the failure is a CORS error with no server-side trace.
	Origins []string `toml:"origins"`
}

// Store is where models are read from. Roots are searched in order and the
// first one holding a name serves it, so a root listed earlier shadows a copy
// of the same model further down. A root with a manifests/ directory is read
// as an ollama store, anything else as a directory of GGUF files. Leaving this
// empty searches ~/models and then ollama's own root.
type Store struct {
	Roots []string `toml:"roots"`
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

// StringList is a list that a config file may also write as one comma
// separated string, matching how llama.cpp takes --device.
type StringList []string

func (l *StringList) UnmarshalTOML(v any) error {
	switch t := v.(type) {
	case string:
		*l = splitList(t)
	case []any:
		out := make([]string, 0, len(t))
		for _, e := range t {
			s, ok := e.(string)
			if !ok {
				return fmt.Errorf("expected a string or list of strings, got %T in the list", e)
			}
			out = append(out, s)
		}
		*l = out
	default:
		return fmt.Errorf("expected a string or list of strings, got %T", v)
	}
	return nil
}

func splitList(s string) []string {
	var out []string
	for _, e := range strings.Split(s, ",") {
		if e = strings.TrimSpace(e); e != "" {
			out = append(out, e)
		}
	}
	return out
}

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
	// AllowPartialOffload permits a deliberately bounded -ngl profile to leave
	// whole layers on the CPU. False/unset retains the strict fit guarantee.
	AllowPartialOffload *bool `toml:"allow_partial_offload"`
	// GPUVRAMCapMiB is a fail-closed post-load ceiling measured from the child
	// process' DRM fdinfo, including caches and graphs as well as weights.
	GPUVRAMCapMiB *int    `toml:"gpu_vram_cap_mib"`
	Parallel      *int    `toml:"parallel"`
	FlashAttn     *string `toml:"flash_attn"`
	Projector     *bool   `toml:"projector"`
	Backend       *string `toml:"backend"`
	// NumCPUMoE and OverrideTensor place chosen tensors on the CPU on purpose.
	// This is not the spill the fit check refuses: "offloaded N/M layers" counts
	// -ngl against the layer count and ignores per-tensor placement, so a load
	// steered here still reports N/N and still has to.
	NumCPUMoE      *int     `toml:"num_cpu_moe"`
	OverrideTensor []string `toml:"override_tensor"`
	// RPCServers are llama.cpp RPC backend endpoints ("host:port"), each one
	// another device the layer split can land on. rpc-server has no auth or
	// encryption, so these must be LAN-only.
	RPCServers []string `toml:"rpc_servers"`
	// Device, TensorSplit, SplitMode and MainGPU steer which local GPUs the
	// weights land on. llama.cpp keeps each layer's KV cache on the GPU holding
	// that layer, so this is the only way to move KV between cards;
	// NoKVOffload moves it to host RAM instead.
	Device      StringList `toml:"device"`
	TensorSplit *string    `toml:"tensor_split"`
	SplitMode   *string    `toml:"split_mode"`
	MainGPU     *int       `toml:"main_gpu"`
	NoKVOffload *bool      `toml:"no_kv_offload"`
	// MoEExpertCache and MoEExpertCacheInserts are llama.cpp PR #27861, an
	// unmerged draft: a VRAM LRU over host-resident experts, decode-only, and
	// pointless without NumCPUMoE or OverrideTensor putting experts there.
	MoEExpertCache        *int `toml:"moe_expert_cache"`
	MoEExpertCacheInserts *int `toml:"moe_expert_cache_inserts"`
	// KVStreamArenaMiB keeps the authoritative KV cache in pinned host RAM and
	// streams pages through a VRAM arena of this size, which buys context a card
	// could not otherwise hold. Zero or unset is llama.cpp's normal behaviour.
	// The MTP draft context does not stream and shares no pool with it, so a
	// draft cache still needs its own VRAM on top of the arena.
	KVStreamArenaMiB *int `toml:"kv_stream_arena_mib"`
	// Embeddings restricts the process to the embedding endpoints.
	// llama-server refuses /v1/embeddings without it, and refuses generation
	// with it, so it is process-level and normally detected from the model.
	Embeddings *bool `toml:"embeddings"`
	// Pooling is how token embeddings are reduced to one vector. Left unset,
	// the model's own choice applies, which is what a dedicated embedding model
	// wants; a causal model has none, and llama.cpp's OpenAI endpoint rejects
	// "none" outright.
	Pooling *string `toml:"pooling"`

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

	// The rest of ollama's sampler options. llama-server accepts every one of
	// these under the same name, so dropping them silently was a divergence
	// that could only be found by measuring the output.
	Mirostat         *int     `toml:"mirostat"`
	MirostatTau      *float32 `toml:"mirostat_tau"`
	MirostatEta      *float32 `toml:"mirostat_eta"`
	PresencePenalty  *float32 `toml:"presence_penalty"`
	FrequencyPenalty *float32 `toml:"frequency_penalty"`
	RepeatLastN      *int     `toml:"repeat_last_n"`
	TypicalP         *float32 `toml:"typical_p"`
	NumKeep          *int     `toml:"num_keep"`

	KeepAlive *string `toml:"keep_alive"`
}

// Default is the configuration used when no file exists. It encodes the machine
// this was built for: ollama's llama.cpp build on the ROCm backend, and the
// settings the benchmarks in gfx1200-lab showed to be worth having.
func Default() Config {
	return Config{
		Server: Server{Listen: "127.0.0.1:11435"},
		Llama:  Llama{LibDir: "/usr/local/lib/ollama", Backend: "rocm_v7_2"},
		WoL:    map[string]string{},
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
	if len(file.Store.Roots) > 0 {
		cfg.Store.Roots = expandRoots(file.Store.Roots)
	}
	cfg.Defaults = Merge(cfg.Defaults, file.Defaults)
	for name, p := range file.Models {
		cfg.Models[name] = p
	}
	for addr, mac := range file.WoL {
		cfg.WoL[addr] = mac
	}

	if err := cfg.Validate(); err != nil {
		return cfg, fmt.Errorf("%s: %w", path, err)
	}
	return cfg, nil
}

// expandRoots resolves a leading ~, which a config file is the natural place
// to write and which no shell expands on alpakka's behalf.
func expandRoots(roots []string) []string {
	home, err := os.UserHomeDir()
	out := make([]string, 0, len(roots))
	for _, r := range roots {
		if err == nil && (r == "~" || strings.HasPrefix(r, "~/")) {
			r = filepath.Join(home, strings.TrimPrefix(r[1:], "/"))
		}
		out = append(out, r)
	}
	return out
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
	setIf(&out.AllowPartialOffload, over.AllowPartialOffload)
	setIf(&out.GPUVRAMCapMiB, over.GPUVRAMCapMiB)
	setIf(&out.Parallel, over.Parallel)
	setIf(&out.FlashAttn, over.FlashAttn)
	setIf(&out.Projector, over.Projector)
	setIf(&out.Backend, over.Backend)
	setIf(&out.NumCPUMoE, over.NumCPUMoE)
	setIf(&out.MoEExpertCache, over.MoEExpertCache)
	setIf(&out.MoEExpertCacheInserts, over.MoEExpertCacheInserts)
	setIf(&out.KVStreamArenaMiB, over.KVStreamArenaMiB)
	setIf(&out.Embeddings, over.Embeddings)
	setIf(&out.Pooling, over.Pooling)
	setIf(&out.ReasoningEffort, over.ReasoningEffort)
	setIf(&out.Temperature, over.Temperature)
	setIf(&out.TopK, over.TopK)
	setIf(&out.TopP, over.TopP)
	setIf(&out.MinP, over.MinP)
	setIf(&out.RepeatPenalty, over.RepeatPenalty)
	setIf(&out.Seed, over.Seed)
	setIf(&out.NumPredict, over.NumPredict)
	setIf(&out.Mirostat, over.Mirostat)
	setIf(&out.MirostatTau, over.MirostatTau)
	setIf(&out.MirostatEta, over.MirostatEta)
	setIf(&out.PresencePenalty, over.PresencePenalty)
	setIf(&out.FrequencyPenalty, over.FrequencyPenalty)
	setIf(&out.RepeatLastN, over.RepeatLastN)
	setIf(&out.TypicalP, over.TypicalP)
	setIf(&out.NumKeep, over.NumKeep)
	setIf(&out.KeepAlive, over.KeepAlive)
	if over.Stop != nil {
		out.Stop = over.Stop
	}
	if over.OverrideTensor != nil {
		out.OverrideTensor = over.OverrideTensor
	}
	if over.RPCServers != nil {
		out.RPCServers = over.RPCServers
	}
	if over.Device != nil {
		out.Device = over.Device
	}
	setIf(&out.TensorSplit, over.TensorSplit)
	setIf(&out.SplitMode, over.SplitMode)
	setIf(&out.MainGPU, over.MainGPU)
	setIf(&out.NoKVOffload, over.NoKVOffload)
	return out
}

func setIf[T any](dst **T, src *T) {
	if src != nil {
		*dst = src
	}
}

func ptr[T any](v T) *T { return &v }

// ValidPoolings are the reductions llama.cpp implements.
var ValidPoolings = []string{"none", "mean", "cls", "last", "rank"}

// ValidSplitModes are the values llama-server's --split-mode accepts.
var ValidSplitModes = []string{"none", "layer", "row", "tensor"}

var reTensorSplit = regexp.MustCompile(`^[0-9]+(\.[0-9]+)?(,[0-9]+(\.[0-9]+)?)*$`)

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
	for addr, mac := range c.WoL {
		hw, err := net.ParseMAC(mac)
		if err != nil {
			return fmt.Errorf("wol.%q: %w", addr, err)
		}
		if len(hw) != 6 {
			return fmt.Errorf("wol.%q: %q must be a 6-byte MAC address", addr, mac)
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
	if p.Pooling != nil && !valid(*p.Pooling, ValidPoolings) {
		return fmt.Errorf("pooling %q: must be one of %v", *p.Pooling, ValidPoolings)
	}
	if p.NumCtx != nil && *p.NumCtx <= 0 {
		return fmt.Errorf("num_ctx must be positive, got %d", *p.NumCtx)
	}
	if p.AllowPartialOffload != nil && *p.AllowPartialOffload {
		if p.NumGPU == nil || *p.NumGPU < 0 || *p.NumGPU >= 99 {
			return fmt.Errorf("allow_partial_offload requires an explicit num_gpu between 0 and 98")
		}
	}
	if p.GPUVRAMCapMiB != nil && *p.GPUVRAMCapMiB <= 0 {
		return fmt.Errorf("gpu_vram_cap_mib must be positive, got %d", *p.GPUVRAMCapMiB)
	}
	if p.KVStreamArenaMiB != nil && *p.KVStreamArenaMiB > 0 &&
		p.Parallel != nil && *p.Parallel != 1 {
		return fmt.Errorf("kv_stream_arena_mib streams a single sequence and needs --parallel 1, got parallel = %d",
			*p.Parallel)
	}
	if p.SplitMode != nil && !valid(*p.SplitMode, ValidSplitModes) {
		return fmt.Errorf("split_mode %q: must be one of %v", *p.SplitMode, ValidSplitModes)
	}
	if p.TensorSplit != nil && !reTensorSplit.MatchString(*p.TensorSplit) {
		return fmt.Errorf("tensor_split %q: must be comma-separated proportions such as \"12,6\"", *p.TensorSplit)
	}
	if p.MainGPU != nil && *p.MainGPU < 0 {
		return fmt.Errorf("main_gpu must not be negative, got %d", *p.MainGPU)
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
//
// The json tags name each field the way the request options object does, so a
// benchmark result reads back in the vocabulary it was asked in.
type Runtime struct {
	Model               string `json:"model"` // canonical model name, for logs and /api/ps
	ModelPath           string `json:"model_path"`
	ProjectorPath       string `json:"projector_path"` // empty when the projector is disabled or absent
	NumCtx              int    `json:"num_ctx"`
	CacheTypeK          string `json:"cache_type_k"`
	CacheTypeV          string `json:"cache_type_v"`
	SpecType            string `json:"spec_type"`
	SpecDraftNMax       int    `json:"spec_draft_n_max"`
	SpecDraftNMin       int    `json:"spec_draft_n_min"`
	NumGPU              int    `json:"num_gpu"`
	AllowPartialOffload bool   `json:"allow_partial_offload"`
	GPUVRAMCapMiB       int    `json:"gpu_vram_cap_mib"`
	Parallel            int    `json:"parallel"`
	FlashAttn           string `json:"flash_attn"`
	Backend             string `json:"backend"`
	NumCPUMoE           int    `json:"num_cpu_moe"`
	// OverrideTensor is the profile's entries joined by commas, which is
	// llama.cpp's own separator for the flag, so nothing an entry could carry is
	// lost by flattening it. A slice here would cost Runtime its comparability
	// and with it the reload check.
	OverrideTensor string `json:"override_tensor"`
	// RPCServers is the profile's entries joined by commas, llama.cpp's own
	// separator for --rpc.
	RPCServers string `json:"rpc_servers"`
	// Device is the profile's entries joined by commas, as --device takes them.
	Device      string `json:"device"`
	TensorSplit string `json:"tensor_split"`
	SplitMode   string `json:"split_mode"`
	// MainGPU is -1 when unset, because 0 is a real device index.
	MainGPU     int  `json:"main_gpu"`
	NoKVOffload bool `json:"no_kv_offload"`
	// MoEExpertCache is llama.cpp PR #27861's --moe-expert-cache, absent from
	// every released build.
	MoEExpertCache        int `json:"moe_expert_cache"`
	MoEExpertCacheInserts int `json:"moe_expert_cache_inserts"`
	// KVStreamArenaMiB is llama.cpp's --kv-stream-arena-mib, zero when the KV
	// cache lives in VRAM as usual.
	KVStreamArenaMiB int `json:"kv_stream_arena_mib"`
	// Embedding starts llama-server with --embeddings. It is part of Runtime
	// because llama-server cannot serve both generation and embeddings from
	// one process: switching between the two has to reload.
	Embedding bool `json:"embeddings"`
	// Pooling is llama.cpp's --pooling, empty for the model's own default.
	Pooling string `json:"pooling"`
}

// Runtime resolves the profile's process-level settings for a model.
// projectorPath is the projector the store found; the profile can suppress it,
// which on a 16 GB card is the difference between fitting and not. embedding is
// what the store detected the model to be; the profile can override that too.
func (p Profile) Runtime(model, modelPath, projectorPath string, embedding bool) Runtime {
	rt := Runtime{
		Model:                 model,
		ModelPath:             modelPath,
		ProjectorPath:         projectorPath,
		NumCtx:                deref(p.NumCtx, 32768),
		CacheTypeK:            deref(p.CacheTypeK, "q8_0"),
		CacheTypeV:            deref(p.CacheTypeV, "q8_0"),
		SpecType:              deref(p.SpecType, ""),
		SpecDraftNMax:         deref(p.SpecDraftNMax, 0),
		SpecDraftNMin:         deref(p.SpecDraftNMin, 0),
		NumGPU:                deref(p.NumGPU, 99),
		AllowPartialOffload:   deref(p.AllowPartialOffload, false),
		GPUVRAMCapMiB:         deref(p.GPUVRAMCapMiB, 0),
		Parallel:              deref(p.Parallel, 1),
		FlashAttn:             deref(p.FlashAttn, "on"),
		Backend:               deref(p.Backend, ""),
		NumCPUMoE:             deref(p.NumCPUMoE, 0),
		OverrideTensor:        strings.Join(p.OverrideTensor, ","),
		RPCServers:            strings.Join(p.RPCServers, ","),
		Device:                strings.Join(p.Device, ","),
		TensorSplit:           strings.ReplaceAll(deref(p.TensorSplit, ""), " ", ""),
		SplitMode:             deref(p.SplitMode, ""),
		MainGPU:               deref(p.MainGPU, -1),
		NoKVOffload:           deref(p.NoKVOffload, false),
		MoEExpertCache:        deref(p.MoEExpertCache, 0),
		MoEExpertCacheInserts: deref(p.MoEExpertCacheInserts, 0),
		KVStreamArenaMiB:      deref(p.KVStreamArenaMiB, 0),
		Embedding:             embedding,
		Pooling:               deref(p.Pooling, ""),
	}
	if p.Embeddings != nil {
		rt.Embedding = *p.Embeddings
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
