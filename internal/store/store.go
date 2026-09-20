// Package store resolves models from disk.
//
// A Source is a place models come from; this file implements the one that
// reads ollama's blob and manifest layout. It is strictly read-only: alpakka
// resolves models another tool has already fetched and never writes here.
package store

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/types/model"

	"github.com/oswald/alpakka/internal/gguf"
)

// Layer media types used by ollama's manifests.
const (
	MediaModel     = "application/vnd.ollama.image.model"
	MediaTemplate  = "application/vnd.ollama.image.template"
	MediaSystem    = "application/vnd.ollama.image.system"
	MediaParams    = "application/vnd.ollama.image.params"
	MediaProjector = "application/vnd.ollama.image.projector"
	MediaLicense   = "application/vnd.ollama.image.license"
	MediaAdapter   = "application/vnd.ollama.image.adapter"
)

// DefaultRoot is where ollama keeps its models, honouring OLLAMA_MODELS.
func DefaultRoot() string {
	if v := os.Getenv("OLLAMA_MODELS"); v != "" {
		return v
	}
	return "/var/lib/ollama/.ollama/models"
}

// Store is a read-only view of an ollama model store.
type Store struct {
	root  string
	cache *ggufCache
}

// New opens the store rooted at dir, which contains blobs/ and manifests/.
func New(root string) *Store {
	return &Store{root: root, cache: newGGUFCache()}
}

// Root returns the store's root directory.
func (s *Store) Root() string { return s.root }

type manifest struct {
	SchemaVersion int     `json:"schemaVersion"`
	Config        Layer   `json:"config"`
	Layers        []Layer `json:"layers"`
}

// Config is ollama's config blob. It is the source of truth for the details
// ollama reports: two manifests can point at the same weights blob and still
// report different quantization levels, because this is where that is recorded.
type Config struct {
	ModelFormat   string   `json:"model_format"`
	ModelFamily   string   `json:"model_family"`
	ModelFamilies []string `json:"model_families"`
	ModelType     string   `json:"model_type"`
	FileType      string   `json:"file_type"`
	Renderer      string   `json:"renderer"`
	Parser        string   `json:"parser"`
	Capabilities  []string `json:"capabilities"`
}

// Layer is one entry of a manifest.
type Layer struct {
	MediaType string `json:"mediaType"`
	Digest    string `json:"digest"`
	Size      int64  `json:"size"`
	From      string `json:"from,omitempty"`
}

// Model is a model resolved from a Source.
type Model struct {
	Name        string // canonical "name:tag", as ollama reports it
	Digest      string // stable id, matching /api/tags
	Size        int64  // every file the model is made of
	ModifiedAt  time.Time
	ParentModel string

	ModelPath     string // file holding the GGUF weights
	ProjectorPath string // vision projector, empty when the model has none
	Config        Config
	Template      string
	System        string
	License       string
	Params        map[string]any

	// goTemplate says Template came from ollama's template layer rather than
	// the jinja one in the GGUF, which decides whether the Go template
	// variables are worth sniffing for capabilities.
	goTemplate bool
	cache      *ggufCache
}

// blobPath maps a "sha256:abc" digest to its file in blobs/.
func (s *Store) blobPath(digest string) string {
	return filepath.Join(s.root, "blobs", strings.Replace(digest, ":", "-", 1))
}

// canonicalName renders a manifest path the way ollama names the model:
// library models drop their registry and namespace, other registry.ollama.ai
// models keep the namespace, and everything else keeps its full host path.
func canonicalName(rel string) (string, bool) {
	parts := strings.Split(filepath.ToSlash(rel), "/")
	if len(parts) < 4 {
		return "", false
	}
	registry, namespace := parts[0], parts[1]
	name := strings.Join(parts[2:len(parts)-1], "/")
	tag := parts[len(parts)-1]

	switch {
	case registry == "registry.ollama.ai" && namespace == "library":
		return name + ":" + tag, true
	case registry == "registry.ollama.ai":
		return namespace + "/" + name + ":" + tag, true
	default:
		return registry + "/" + namespace + "/" + name + ":" + tag, true
	}
}

// List returns every model in the store, newest first, matching /api/tags order.
func (s *Store) List() ([]Model, error) {
	dir := filepath.Join(s.root, "manifests")
	var models []Model

	err := filepath.WalkDir(dir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			return nil
		}
		rel, err := filepath.Rel(dir, path)
		if err != nil {
			return err
		}
		name, ok := canonicalName(rel)
		if !ok {
			return nil
		}
		m, err := s.load(path, name)
		if err != nil {
			// A manifest that does not parse, or whose blobs were garbage
			// collected, should not take down the whole listing.
			return nil
		}
		models = append(models, *m)
		return nil
	})
	if err != nil {
		return nil, err
	}

	sortNewestFirst(models)
	return models, nil
}

// Get resolves a model by name. The tag defaults to "latest".
//
// The name maps directly onto a manifest path, so the common case is one open
// rather than a walk over the whole store. Every request resolves a model, and
// /api/ps is polled, so the difference is between a handful of syscalls and
// re-reading every manifest and config blob on the machine.
func (s *Store) Get(name string) (*Model, error) {
	want := name
	if !strings.Contains(path(want), ":") {
		want += ":latest"
	}

	if rel, ok := manifestRel(want); ok {
		if canonical, ok := canonicalName(rel); ok && canonical == want {
			if m, err := s.load(filepath.Join(s.root, "manifests", rel), canonical); err == nil {
				return m, nil
			}
		}
	}

	// Fall back to a scan: a manifest may sit somewhere the direct mapping does
	// not predict, and being right matters more here than being quick.
	models, err := s.List()
	if err != nil {
		return nil, err
	}
	for i := range models {
		if models[i].Name == want {
			return &models[i], nil
		}
	}
	return nil, fmt.Errorf("%q: %w", name, ErrNotFound)
}

// manifestRel is canonicalName in reverse: it turns "qwen3:0.6b" back into
// "registry.ollama.ai/library/qwen3/0.6b". It reports false for anything that
// could escape the store, since the name arrives from an HTTP request.
func manifestRel(name string) (string, bool) {
	repo, tag := name, ""
	p := path(name)
	i := strings.LastIndex(p, ":")
	if i < 0 {
		return "", false
	}
	off := len(name) - len(p)
	repo, tag = name[:off+i], name[off+i+1:]

	parts := strings.Split(repo, "/")
	switch len(parts) {
	case 1:
		parts = append([]string{"registry.ollama.ai", "library"}, parts...)
	case 2:
		parts = append([]string{"registry.ollama.ai"}, parts...)
	}
	parts = append(parts, tag)

	for _, part := range parts {
		if part == "" || part == "." || part == ".." || strings.ContainsAny(part, `/\`) {
			return "", false
		}
	}
	return strings.Join(parts, "/"), true
}

// path strips any registry host so a colon in "host:port" is not mistaken for
// a tag separator.
func path(name string) string {
	if i := strings.Index(name, "/"); i >= 0 {
		return name[i:]
	}
	return name
}

func (s *Store) load(manifestPath, name string) (*Model, error) {
	raw, err := os.ReadFile(manifestPath)
	if err != nil {
		return nil, err
	}
	var mf manifest
	if err := json.Unmarshal(raw, &mf); err != nil {
		return nil, err
	}
	info, err := os.Stat(manifestPath)
	if err != nil {
		return nil, err
	}

	var cfg Config
	if body, err := s.readBlob(mf.Config.Digest); err == nil {
		_ = json.Unmarshal([]byte(body), &cfg)
	}

	sum := sha256.Sum256(raw)
	m := &Model{
		Name:       name,
		Digest:     hex.EncodeToString(sum[:]),
		Size:       mf.Config.Size,
		ModifiedAt: info.ModTime(),
		Config:     cfg,
		Params:     map[string]any{},
		cache:      s.cache,
	}

	for _, l := range mf.Layers {
		m.Size += l.Size
		if m.ParentModel == "" && l.From != "" {
			m.ParentModel = l.From
		}
		switch l.MediaType {
		case MediaModel:
			m.ModelPath = s.blobPath(l.Digest)
		case MediaProjector:
			m.ProjectorPath = s.blobPath(l.Digest)
		case MediaTemplate:
			m.Template, _ = s.readBlob(l.Digest)
			m.goTemplate = m.Template != ""
		case MediaSystem:
			m.System, _ = s.readBlob(l.Digest)
		case MediaLicense:
			m.License, _ = s.readBlob(l.Digest)
		case MediaParams:
			if body, err := s.readBlob(l.Digest); err == nil {
				_ = json.Unmarshal([]byte(body), &m.Params)
			}
		}
	}
	if m.ModelPath == "" {
		return nil, fmt.Errorf("manifest %q has no model layer", name)
	}
	if _, err := os.Stat(m.ModelPath); err != nil {
		return nil, fmt.Errorf("model %q: %w", name, err)
	}
	return m, nil
}

func (s *Store) readBlob(digest string) (string, error) {
	f, err := os.Open(s.blobPath(digest))
	if err != nil {
		return "", err
	}
	defer f.Close()
	// Template and system blobs are small; cap the read so a mislabelled
	// weights layer cannot pull gigabytes into memory.
	b, err := io.ReadAll(io.LimitReader(f, 1<<20))
	return string(b), err
}

// GGUF parses and caches the model's GGUF metadata header.
func (m *Model) GGUF() (*gguf.File, error) {
	if m.cache == nil {
		return gguf.Open(m.ModelPath)
	}
	return m.cache.open(m.ModelPath)
}

// Details renders the details object ollama reports in /api/tags, /api/show and
// /api/ps. Family and quantization come from the config blob; the two lengths
// are only in the GGUF header, and are omitted if it cannot be read.
func (m *Model) Details() api.ModelDetails {
	d := api.ModelDetails{
		ParentModel:       m.ParentModel,
		Format:            m.Config.ModelFormat,
		Family:            m.Config.ModelFamily,
		Families:          m.Config.ModelFamilies,
		ParameterSize:     m.Config.ModelType,
		QuantizationLevel: m.Config.FileType,
	}
	f, err := m.GGUF()
	if err != nil {
		return d
	}
	if n, ok := f.ArchUint("context_length"); ok {
		d.ContextLength = int(n)
	}
	if n, ok := f.ArchUint("embedding_length"); ok {
		d.EmbeddingLength = int(n)
	}
	return d
}

// Capabilities reports what the model can do.
//
// This follows ollama's /api/show, which computes capabilities live from the
// model itself. Ollama's /api/tags answers from a cache built at pull time and
// the two genuinely disagree — /api/show reports audio and vision for
// gemma4:e4b where /api/tags reports neither. /api/show is the endpoint clients
// query to decide whether to send tools or ask for thinking, so it is the one
// worth matching.
func (m *Model) Capabilities() []model.Capability {
	var caps []model.Capability
	add := func(c model.Capability) {
		for _, existing := range caps {
			if existing == c {
				return
			}
		}
		caps = append(caps, c)
	}

	for _, c := range m.Config.Capabilities {
		add(model.Capability(c))
	}

	f, err := m.GGUF()
	if err != nil {
		add(model.CapabilityCompletion)
	} else {
		// The GGUF's own chat template is what llama.cpp will actually apply,
		// so it is the honest source for what the model can be asked to do.
		tmpl, _ := f.String("tokenizer.chat_template")
		if chatTemplateHasTools(tmpl) {
			add(model.CapabilityTools)
		}
		if chatTemplateHasThinking(tmpl) {
			add(model.CapabilityThinking)
		}

		if _, ok := f.KV["pooling_type"]; ok {
			add(model.CapabilityEmbedding)
		} else {
			add(model.CapabilityCompletion)
		}
		// These live under the architecture namespace, e.g. gemma4.vision.block_count.
		if _, ok := f.ArchKV("vision.block_count"); ok {
			add(model.CapabilityVision)
		}
		if _, ok := f.ArchKV("audio.block_count"); ok {
			add(model.CapabilityAudio)
		}
	}

	if m.ProjectorPath != "" {
		add(model.CapabilityVision)
	}

	if m.goTemplate {
		// Go templates expose these as template variables.
		if strings.Contains(m.Template, "Tools") {
			add(model.CapabilityTools)
		}
		if strings.Contains(m.Template, ".Thinking") {
			add(model.CapabilityThinking)
		}
	}

	if tools, thinking, ok := parserCapabilities(m.Config.Parser); ok {
		if tools {
			add(model.CapabilityTools)
		}
		if thinking {
			add(model.CapabilityThinking)
		}
	}

	if m.Config.ModelFamily == "gptoss" || m.Config.ModelFamily == "gpt-oss" {
		add(model.CapabilityThinking)
	}

	return caps
}

// parserCapabilities reports what one of ollama's built-in parsers supports.
//
// Ollama keeps this as a registry of parser implementations that changes with
// every release. Rather than mirror that registry — the kind of permanent
// rebase tax this project exists to avoid — this covers the parsers that models
// in the store actually declare. An unknown parser reports nothing and the
// model's own chat template decides, which is the right fallback: the template
// is what llama.cpp will apply regardless.
func parserCapabilities(name string) (tools, thinking, known bool) {
	switch name {
	case "":
		return false, false, false
	case "qwen3.5", "ornith", "qwen3-coder", "gemma4", "harmony", "deepseek3",
		"cogito", "glm-4.7", "olmo3-think", "lfm2-thinking", "qwen3-vl-thinking":
		return true, true, true
	case "qwen3", "qwen3-vl-instruct", "ministral", "olmo3", "lfm2", "functiongemma":
		return true, false, true
	case "passthrough":
		return false, false, true
	}
	return false, false, false
}

// usesOllamaRenderedChat reports whether ollama would render the chat itself
// rather than handing the messages to the model's own jinja template.
func (m *Model) usesOllamaRenderedChat() bool {
	return m.Config.Renderer != "" || m.Config.Parser != "" || m.goTemplate
}

func chatTemplateHasTools(tmpl string) bool {
	if tmpl == "" {
		return false
	}
	return strings.Contains(tmpl, "tools") || strings.Contains(tmpl, "tool_call")
}

func chatTemplateHasThinking(tmpl string) bool {
	if tmpl == "" {
		return false
	}
	if strings.Contains(tmpl, "<think>") && strings.Contains(tmpl, "</think>") {
		return true
	}
	// Some Qwen and DeepSeek templates strip earlier reasoning by splitting
	// assistant content on the closing tag; reasoning is still extractable.
	return (strings.Contains(tmpl, "content.split('</think>')") ||
		strings.Contains(tmpl, `content.split("</think>")`)) &&
		!strings.Contains(tmpl, "reasoning_content") &&
		!strings.Contains(tmpl, "<SPECIAL_12>")
}

// IsEmbedding reports whether the model produces embeddings rather than text.
//
// llama-server has to be started with --embeddings for one and without it for
// the other, so this decides a process-level flag, not a request field.
func (m *Model) IsEmbedding() bool {
	for _, c := range m.Capabilities() {
		if c == model.CapabilityEmbedding {
			return true
		}
	}
	return false
}

// DeclaresPooling reports whether the GGUF names its own pooling type, which
// dedicated embedding models do and causal models do not.
func (m *Model) DeclaresPooling() bool {
	f, err := m.GGUF()
	if err != nil {
		return false
	}
	_, ok := f.KV["pooling_type"]
	return ok
}

// TrainedContext is the context length the model was trained for, or zero when
// the GGUF header does not say.
func (m *Model) TrainedContext() int {
	f, err := m.GGUF()
	if err != nil {
		return 0
	}
	if n, ok := f.ArchUint("context_length"); ok {
		return int(n)
	}
	return 0
}

// ListResponse renders the model as an /api/tags entry.
func (m *Model) ListResponse() api.ListModelResponse {
	return api.ListModelResponse{
		Name:         m.Name,
		Model:        m.Name,
		ModifiedAt:   m.ModifiedAt,
		Size:         m.Size,
		Digest:       m.Digest,
		Details:      m.Details(),
		Capabilities: m.Capabilities(),
	}
}
