package store

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
)

// An ambiguous name is an answer, not a miss, so it must not send the lookup on
// to the next root.
var ErrNotFound = errors.New("model not found")

func DefaultRoots() []string {
	home, err := os.UserHomeDir()
	if err != nil {
		return nil
	}
	return []string{filepath.Join(home, "models")}
}

type Multi struct {
	sources []Source
	logf    func(string, ...any)

	mu     sync.Mutex
	warned map[string]bool
}

func NewMulti(logf func(string, ...any), sources ...Source) *Multi {
	if logf == nil {
		logf = func(string, ...any) {}
	}
	return &Multi{sources: sources, logf: logf, warned: map[string]bool{}}
}

func (m *Multi) List() ([]Model, error) {
	var out []Model
	seen := map[string]bool{}
	var firstErr error

	for _, s := range m.sources {
		models, err := s.List()
		if err != nil {
			// One unreadable root should not blank out /api/tags for the rest.
			if firstErr == nil {
				firstErr = err
			}
			continue
		}
		for _, model := range models {
			if seen[model.Name] {
				m.warnShadowed(model)
				continue
			}
			seen[model.Name] = true
			out = append(out, model)
		}
	}
	if out == nil && firstErr != nil {
		return nil, firstErr
	}

	sortNewestFirst(out)
	return out, nil
}

// Only a miss falls through. A root that cannot decide which tag was meant has
// answered, and searching on would serve a different model.
func (m *Multi) Get(name string) (*Model, error) {
	for _, s := range m.sources {
		model, err := s.Get(name)
		if err == nil {
			return model, nil
		}
		if !errors.Is(err, ErrNotFound) {
			return nil, err
		}
	}
	return nil, fmt.Errorf("%q: %w", name, ErrNotFound)
}

func (m *Multi) warnShadowed(model Model) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.warned[model.Name] {
		return
	}
	m.warned[model.Name] = true
	m.logf("model %s: shadowed copy at %s is not served", model.Name, model.ModelPath)
}

var _ Source = (*Multi)(nil)
