// Package store resolves models from disk, read-only.
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

const (
	MediaModel     = "application/vnd.ollama.image.model"
	MediaTemplate  = "application/vnd.ollama.image.template"
	MediaSystem    = "application/vnd.ollama.image.system"
	MediaParams    = "application/vnd.ollama.image.params"
	MediaProjector = "application/vnd.ollama.image.projector"
	MediaLicense   = "application/vnd.ollama.image.license"
	MediaAdapter   = "application/vnd.ollama.image.adapter"
)

func DefaultRoot() string {
	if v := os.Getenv("OLLAMA_MODELS"); v != "" {
		return v
	}
	return "/var/lib/ollama/.ollama/models"
}

type Store struct {
	root  string
	cache *ggufCache
}

func New(root string) *Store {
	return &Store{root: root, cache: newGGUFCache()}
}

func (s *Store) Root() string { return s.root }

type manifest struct {
	SchemaVersion int     `json:"schemaVersion"`
	Config        Layer   `json:"config"`
	Layers        []Layer `json:"layers"`
}

// The source of truth for ollama's reported details: two manifests can share a
// weights blob and still report different quantization.
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

type Layer struct {
	MediaType string `json:"mediaType"`
	Digest    string `json:"digest"`
	Size      int64  `json:"size"`
	From      string `json:"from,omitempty"`
}

type Model struct {
	Name        string
	Digest      string
	Size        int64
	ModifiedAt  time.Time
	ParentModel string

	ModelPath     string
	ProjectorPath string
	Config        Config
	Template      string
	System        string
	License       string
	Params        map[string]any

	// Template came from ollama's template layer rather than the GGUF's jinja, which
	// decides whether Go template variables are worth sniffing for capabilities.
	goTemplate bool
	cache      *ggufCache
}

func (s *Store) blobPath(digest string) string {
	return filepath.Join(s.root, "blobs", strings.Replace(digest, ":", "-", 1))
}

// Library models drop registry and namespace, other registry.ollama.ai models keep
// the namespace, and everything else keeps its full host path.
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
			// A manifest that does not parse or lost its blobs should not take down the listing.
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

// The name maps onto a manifest path, so the common case is one open rather than a
// walk over every manifest and config blob, which matters since /api/ps is polled.
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

	// Fall back to a scan since a manifest may sit where the mapping does not predict.
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

// The name arrives from an HTTP request, so this reports false for anything that
// could escape the store.
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

// Strips any registry host so a colon in host:port is not mistaken for a tag separator.
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
	// Template and system blobs are small, so cap the read against a mislabelled weights layer.
	b, err := io.ReadAll(io.LimitReader(f, 1<<20))
	return string(b), err
}

func (m *Model) GGUF() (*gguf.File, error) {
	if m.cache == nil {
		return gguf.Open(m.ModelPath)
	}
	return m.cache.open(m.ModelPath)
}

// The two lengths are only in the GGUF header and are omitted if it cannot be read.
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

// Follows /api/show, which computes live. /api/tags answers from a pull-time cache and
// disagrees (gemma4:e4b), and clients query /api/show before sending tools or asking for thinking.
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
		// The GGUF's own chat template is what llama.cpp applies, so it is the honest source.
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

// Ollama's parser registry changes every release, so this covers only the parsers
// models in the store declare. An unknown parser reports nothing and the chat template decides.
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
	// Some Qwen and DeepSeek templates strip earlier reasoning by splitting on the
	// closing tag, so reasoning is still extractable.
	return (strings.Contains(tmpl, "content.split('</think>')") ||
		strings.Contains(tmpl, `content.split("</think>")`)) &&
		!strings.Contains(tmpl, "reasoning_content") &&
		!strings.Contains(tmpl, "<SPECIAL_12>")
}

// Decides a process-level flag: llama-server needs --embeddings for one and not the other.
func (m *Model) IsEmbedding() bool {
	for _, c := range m.Capabilities() {
		if c == model.CapabilityEmbedding {
			return true
		}
	}
	return false
}

// Dedicated embedding models name a pooling type and causal models do not.
func (m *Model) DeclaresPooling() bool {
	f, err := m.GGUF()
	if err != nil {
		return false
	}
	_, ok := f.KV["pooling_type"]
	return ok
}

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
