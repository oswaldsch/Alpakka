package store

import (
	"sync"

	"github.com/oswald/alpakka/internal/gguf"
)

// Source is a place models are read from. Everything the API layer needs is a
// listing and a lookup by name; where the bytes live is the implementation's
// business.
type Source interface {
	List() ([]Model, error)
	Get(name string) (*Model, error)
}

var _ Source = (*Store)(nil)

// ggufCache memoises parsed GGUF headers by file path. A header parse walks the
// whole tensor table, /api/ps is polled and every request resolves a model, so
// the same file would otherwise be re-read many times a minute.
type ggufCache struct {
	mu    sync.Mutex
	files map[string]*gguf.File
}

func newGGUFCache() *ggufCache { return &ggufCache{files: map[string]*gguf.File{}} }

func (c *ggufCache) open(path string) (*gguf.File, error) {
	c.mu.Lock()
	f, ok := c.files[path]
	c.mu.Unlock()
	if ok {
		return f, nil
	}

	f, err := gguf.Open(path)
	if err != nil {
		return nil, err
	}
	c.mu.Lock()
	c.files[path] = f
	c.mu.Unlock()
	return f, nil
}
