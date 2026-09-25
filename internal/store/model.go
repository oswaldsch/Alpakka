// Package store resolves models from a directory of GGUF files, read-only.
package store

import (
	"strings"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/types/model"

	"github.com/oswaldsch/alpakka/internal/gguf"
)

type Model struct {
	Name       string
	Digest     string
	Size       int64
	ModifiedAt time.Time

	ModelPath     string
	ProjectorPath string
	Template      string

	// Every file of the weights, one unless llama.cpp split them. ModelPath is the first.
	Parts []string
	// A projector named for the repo rather than the tag serves every tag beside it.
	ProjectorShared bool

	cache *ggufCache
}

func (m *Model) GGUF() (*gguf.File, error) {
	if m.cache == nil {
		return gguf.Open(m.ModelPath)
	}
	return m.cache.open(m.ModelPath)
}

func (m *Model) Details() api.ModelDetails {
	d := api.ModelDetails{Format: "gguf", QuantizationLevel: "unknown"}
	f, err := m.GGUF()
	if err != nil {
		return d
	}
	if arch := f.Architecture(); arch != "" {
		d.Family = arch
		d.Families = []string{arch}
	}
	d.ParameterSize = gguf.HumanParams(f.ParameterCount())
	if ft, ok := f.Uint("general.file_type"); ok {
		d.QuantizationLevel = gguf.FileTypeName(ft)
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
		// Harmony templates reason in channels, which the <think> check cannot see.
		if f.Architecture() == "gpt-oss" {
			add(model.CapabilityThinking)
		}
	}

	if m.ProjectorPath != "" {
		add(model.CapabilityVision)
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
