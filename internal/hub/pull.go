package hub

import (
	"context"
	"fmt"
	"os"
	"path/filepath"

	"github.com/oswaldsch/alpakka/internal/store"
)

type Target struct {
	Path string
	Repo string
	Size int64
}

func (p *Plan) Targets(root string, withProjector bool) []Target {
	dir := filepath.Join(root, p.Name)
	out := make([]Target, 0, len(p.Weights)+1)

	for i, e := range p.Weights {
		name := p.Tag + ".gguf"
		if len(p.Weights) > 1 {
			part, parts := store.ParseGGUFName(e.Path).Part, len(p.Weights)
			if part == 0 {
				part = i + 1
			}
			name = fmt.Sprintf("%s-%05d-of-%05d.gguf", p.Tag, part, parts)
		}
		out = append(out, Target{Path: filepath.Join(dir, name), Repo: e.Path, Size: e.size()})
	}

	if withProjector && p.Projector != nil {
		out = append(out, Target{
			Path: filepath.Join(dir, p.Tag+".mmproj.gguf"),
			Repo: p.Projector.Path,
			Size: p.Projector.size(),
		})
	}
	return out
}

// An existing tag is left alone unless force is set, since asking for a model that is
// already there is a common mistake that would cost the whole download.
func (c *Client) Pull(ctx context.Context, plan *Plan, root string, withProjector, force bool) error {
	targets := plan.Targets(root, withProjector)

	if !force {
		for _, t := range targets {
			if _, err := os.Stat(t.Path); err == nil {
				return fmt.Errorf("%s already holds %s, pass -f to replace it",
					plan.Name+":"+plan.Tag, t.Path)
			}
		}
	}

	c.Logf("pull %s:%s from %s (%s)", plan.Name, plan.Tag, plan.Repo, human(plan.Size))
	for _, t := range targets {
		c.Logf("pull %s -> %s (%s)", t.Repo, t.Path, human(t.Size))
		url := c.FileURL(plan.Repo, plan.Revision, t.Repo)
		if err := c.Download(ctx, url, t.Path, t.Size); err != nil {
			return err
		}
	}
	c.Logf("pulled %s:%s into %s", plan.Name, plan.Tag, filepath.Join(root, plan.Name))
	return nil
}
