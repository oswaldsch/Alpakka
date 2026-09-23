// Package config holds alpakka's settings and per-model llama-server flag profiles.
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

type Config struct {
	Server Server `toml:"server"`
	Store  Store  `toml:"store"`
	Llama  Llama  `toml:"llama"`
	// Maps an rpc_servers "host:port" to the MAC that gets a Wake-on-LAN packet
	// before a load offloads onto it.
	WoL      map[string]string  `toml:"wol"`
	Defaults Profile            `toml:"defaults"`
	Models   map[string]Profile `toml:"models"`
}

type Server struct {
	Listen string `toml:"listen"`
	// Empty means ollama's default origins. Without them browser clients fail
	// with a CORS error and no server-side trace.
	Origins []string `toml:"origins"`
}

// Roots are searched in order, so an earlier root shadows later ones. A root
// with manifests/ is an ollama store, anything else a GGUF directory.
type Store struct {
	Roots []string `toml:"roots"`
}

type Llama struct {
	LibDir  string `toml:"lib_dir"`
	Backend string `toml:"backend"`
}

func (l Llama) Binary() string { return filepath.Join(l.LibDir, "llama-server") }

// ggml only looks for libggml-hip.so in the executable and working directories,
// otherwise the server silently runs on CPU.
func (l Llama) BackendDir() string { return filepath.Join(l.LibDir, l.Backend) }

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

// Fields are pointers so unset differs from the zero value, letting request
// options layer over model defaults over global defaults.
type Profile struct {
	// Process-level: a change requires a llama-server restart.
	NumCtx     *int    `toml:"num_ctx"`
	CacheTypeK *string `toml:"cache_type_k"`
	CacheTypeV *string `toml:"cache_type_v"`
	// llama.cpp gives the MTP draft context an f16 cache by default, 512 MiB at 128K.
	CacheTypeKDraft *string `toml:"cache_type_k_draft"`
	CacheTypeVDraft *string `toml:"cache_type_v_draft"`
	NumBatch        *int    `toml:"num_batch"`
	NumUBatch       *int    `toml:"num_ubatch"`
	LoadMode        *string `toml:"load_mode"`
	SpecType        *string `toml:"spec_type"`
	SpecDraftNMax   *int    `toml:"spec_draft_n_max"`
	SpecDraftNMin   *int    `toml:"spec_draft_n_min"`
	NumGPU          *int    `toml:"num_gpu"`
	// Lets a bounded -ngl profile leave layers on the CPU. Unset keeps the strict fit check.
	AllowPartialOffload *bool `toml:"allow_partial_offload"`
	// Fail-closed ceiling measured from the child's DRM fdinfo, including caches and graphs.
	GPUVRAMCapMiB *int    `toml:"gpu_vram_cap_mib"`
	Parallel      *int    `toml:"parallel"`
	FlashAttn     *string `toml:"flash_attn"`
	Projector     *bool   `toml:"projector"`
	Backend       *string `toml:"backend"`
	// Deliberate CPU placement, not the spill the fit check refuses. The
	// offloaded N/M count ignores per-tensor placement, so it still reports N/N.
	NumCPUMoE      *int     `toml:"num_cpu_moe"`
	OverrideTensor []string `toml:"override_tensor"`
	// rpc-server has no auth or encryption, so these must be LAN-only.
	RPCServers []string `toml:"rpc_servers"`
	// llama.cpp keeps each layer's KV on the GPU holding that layer, so this is
	// the only way to move KV between cards. NoKVOffload moves it to host RAM.
	Device      StringList `toml:"device"`
	TensorSplit *string    `toml:"tensor_split"`
	SplitMode   *string    `toml:"split_mode"`
	MainGPU     *int       `toml:"main_gpu"`
	NoKVOffload *bool      `toml:"no_kv_offload"`
	// llama.cpp PR #27861, unmerged: a decode-only VRAM LRU over host experts,
	// pointless without NumCPUMoE or OverrideTensor.
	MoEExpertCache        *int `toml:"moe_expert_cache"`
	MoEExpertCacheInserts *int `toml:"moe_expert_cache_inserts"`
	// Streams KV pages from pinned host RAM through a VRAM arena of this size. The
	// MTP draft context is not streamed and needs its own VRAM on top.
	KVStreamArenaMiB *int `toml:"kv_stream_arena_mib"`
	// llama-server refuses /v1/embeddings without it and refuses generation with
	// it, so it is process-level.
	Embeddings *bool `toml:"embeddings"`
	// Unset uses the model's own pooling. A causal model has none and llama.cpp's
	// OpenAI endpoint rejects "none".
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

// Encodes the author's machine: ollama's llama.cpp on ROCm, with the settings
// the gfx1200-lab benchmarks favoured.
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

func (c Config) ForModel(name string) Profile {
	p := c.Defaults
	if over, ok := c.Models[name]; ok {
		p = Merge(p, over)
	}
	// A config may key a model by bare name while requests use name:tag.
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

func Merge(base, over Profile) Profile {
	out := base
	setIf(&out.NumCtx, over.NumCtx)
	setIf(&out.CacheTypeK, over.CacheTypeK)
	setIf(&out.CacheTypeV, over.CacheTypeV)
	setIf(&out.CacheTypeKDraft, over.CacheTypeKDraft)
	setIf(&out.CacheTypeVDraft, over.CacheTypeVDraft)
	setIf(&out.NumBatch, over.NumBatch)
	setIf(&out.NumUBatch, over.NumUBatch)
	setIf(&out.LoadMode, over.LoadMode)
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

var ValidPoolings = []string{"none", "mean", "cls", "last", "rank"}

var ValidLoadModes = []string{"auto", "none", "mmap", "mlock", "mmap+mlock", "dio"}

var ValidSplitModes = []string{"none", "layer", "row", "tensor"}

var reTensorSplit = regexp.MustCompile(`^[0-9]+(\.[0-9]+)?(,[0-9]+(\.[0-9]+)?)*$`)

// Qwen templates reject anything else and silently promote "high" to "xhigh".
// "none" is what turns thinking off, since even "low" yields pages of reasoning and no answer.
var ValidEfforts = []string{"none", "low", "medium", "xhigh"}

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
	if p.NumBatch != nil && *p.NumBatch <= 0 {
		return fmt.Errorf("num_batch must be positive, got %d", *p.NumBatch)
	}
	if p.NumUBatch != nil && *p.NumUBatch <= 0 {
		return fmt.Errorf("num_ubatch must be positive, got %d", *p.NumUBatch)
	}
	if p.LoadMode != nil && !valid(*p.LoadMode, ValidLoadModes) {
		return fmt.Errorf("load_mode %q: must be one of %v", *p.LoadMode, ValidLoadModes)
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

// Comparable on purpose: the supervisor reloads when two Runtimes differ.
// The json tags match the request options vocabulary.
type Runtime struct {
	Model           string `json:"model"`
	ModelPath       string `json:"model_path"`
	ProjectorPath   string `json:"projector_path"`
	NumCtx          int    `json:"num_ctx"`
	CacheTypeK      string `json:"cache_type_k"`
	CacheTypeV      string `json:"cache_type_v"`
	CacheTypeKDraft string `json:"cache_type_k_draft"`
	CacheTypeVDraft string `json:"cache_type_v_draft"`
	// Zero or empty leaves llama.cpp's own default.
	NumBatch            int    `json:"num_batch"`
	NumUBatch           int    `json:"num_ubatch"`
	LoadMode            string `json:"load_mode"`
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
	// Joined by commas, llama.cpp's own separator, so nothing is lost. A slice
	// would cost Runtime its comparability.
	OverrideTensor string `json:"override_tensor"`
	RPCServers     string `json:"rpc_servers"`
	Device         string `json:"device"`
	TensorSplit    string `json:"tensor_split"`
	SplitMode      string `json:"split_mode"`
	// -1 when unset, because 0 is a real device index.
	MainGPU               int  `json:"main_gpu"`
	NoKVOffload           bool `json:"no_kv_offload"`
	MoEExpertCache        int  `json:"moe_expert_cache"`
	MoEExpertCacheInserts int  `json:"moe_expert_cache_inserts"`
	KVStreamArenaMiB      int  `json:"kv_stream_arena_mib"`
	// llama-server cannot serve generation and embeddings from one process, so
	// switching between them reloads.
	Embedding bool   `json:"embeddings"`
	Pooling   string `json:"pooling"`
}

// The profile can suppress the projector, which on a 16 GB card decides whether
// the model fits, and can override the detected embedding flag.
func (p Profile) Runtime(model, modelPath, projectorPath string, embedding bool) Runtime {
	rt := Runtime{
		Model:                 model,
		ModelPath:             modelPath,
		ProjectorPath:         projectorPath,
		NumCtx:                deref(p.NumCtx, 32768),
		CacheTypeK:            deref(p.CacheTypeK, "q8_0"),
		CacheTypeV:            deref(p.CacheTypeV, "q8_0"),
		CacheTypeKDraft:       deref(p.CacheTypeKDraft, ""),
		CacheTypeVDraft:       deref(p.CacheTypeVDraft, ""),
		NumBatch:              deref(p.NumBatch, 0),
		NumUBatch:             deref(p.NumUBatch, 0),
		LoadMode:              deref(p.LoadMode, ""),
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
