package store

import (
	"sync"

	"github.com/oswald/alpakka/internal/gguf"
)

type Source interface {
	List() ([]Model, error)
	Get(name string) (*Model, error)
}

// A header parse walks the whole tensor table, and /api/ps is polled while every
// request resolves a model, so uncached files would be re-read constantly.
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
