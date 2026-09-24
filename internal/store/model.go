// Package store resolves models from a directory of GGUF files, read-only.
package store

import (
	"strings"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/types/model"

	"github.com/oswald/alpakka/internal/gguf"
)

type Config struct {
	ModelFormat   string
	ModelFamily   string
	ModelFamilies []string
	ModelType     string
	FileType      string
}

type Model struct {
	Name       string
	Digest     string
	Size       int64
	ModifiedAt time.Time

	ModelPath     string
	ProjectorPath string
	Config        Config
	Template      string

	cache *ggufCache
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

	if m.Config.ModelFamily == "gpt-oss" {
		add(model.CapabilityThinking)
	}

	return caps
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
