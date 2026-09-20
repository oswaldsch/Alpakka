package hub

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"sort"
	"strings"

	"github.com/oswald/alpakka/internal/store"
)

// DefaultEndpoint is HuggingFace, honouring the variable its own tooling uses.
func DefaultEndpoint() string {
	if v := os.Getenv("HF_ENDPOINT"); v != "" {
		return strings.TrimSuffix(v, "/")
	}
	return "https://huggingface.co"
}

// Client talks to one HuggingFace-shaped endpoint.
type Client struct {
	HTTP     *http.Client
	Endpoint string
	Token    string
	Logf     func(string, ...any)
}

// NewClient reads the token from the environment, which is the only way a
// gated repo can be pulled without prompting for one.
func NewClient(logf func(string, ...any)) *Client {
	token := os.Getenv("HF_TOKEN")
	if token == "" {
		token = os.Getenv("HUGGING_FACE_HUB_TOKEN")
	}
	if logf == nil {
		logf = func(string, ...any) {}
	}
	return &Client{HTTP: http.DefaultClient, Endpoint: DefaultEndpoint(), Token: token, Logf: logf}
}

// Entry is one file in a repo.
type Entry struct {
	Type string `json:"type"`
	Path string `json:"path"`
	Size int64  `json:"size"`
	LFS  *struct {
		Size int64 `json:"size"`
	} `json:"lfs"`
}

// size is the real file size; for an LFS file the outer one may describe the
// pointer rather than the object.
func (e Entry) size() int64 {
	if e.LFS != nil && e.LFS.Size > 0 {
		return e.LFS.Size
	}
	return e.Size
}

// Tree lists the GGUF files in a repo revision.
func (c *Client) Tree(ctx context.Context, repo, revision string) ([]Entry, error) {
	u := fmt.Sprintf("%s/api/models/%s/tree/%s?recursive=1",
		c.Endpoint, url.PathEscape(repo), url.PathEscape(revision))

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return nil, err
	}
	c.auth(req)

	resp, err := c.HTTP.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("listing %s@%s: %s", repo, revision, resp.Status)
	}

	var entries []Entry
	if err := json.NewDecoder(resp.Body).Decode(&entries); err != nil {
		return nil, fmt.Errorf("listing %s@%s: %w", repo, revision, err)
	}

	out := entries[:0]
	for _, e := range entries {
		if e.Type != "directory" && strings.HasSuffix(e.Path, ".gguf") {
			out = append(out, e)
		}
	}
	return out, nil
}

// Plan is what a Ref resolved to: the files to fetch and the name to store
// them under.
type Plan struct {
	Repo      string
	Revision  string
	Name      string
	Tag       string
	Weights   []Entry // in split order; one entry when the model is not split
	Projector *Entry  // the repo's mmproj, whether or not it was asked for
	Size      int64
}

// group is one candidate model in a repo: every part of one quantization.
type group struct {
	tag   string
	parts []Entry
}

// Resolve turns a reference into a concrete set of files to download.
func (c *Client) Resolve(ctx context.Context, ref Ref) (*Plan, error) {
	entries, err := c.Tree(ctx, ref.Repo, ref.Revision)
	if err != nil {
		return nil, err
	}
	if len(entries) == 0 {
		return nil, fmt.Errorf("%s@%s holds no GGUF files", ref.Repo, ref.Revision)
	}

	groups := map[string]*group{}
	var projector *Entry
	for i, e := range entries {
		parsed := store.ParseGGUFName(e.Path)
		if parsed.Projector {
			// The largest projector in a repo is the one to keep: publishers
			// ship F32 beside F16 and the quality difference is free at this
			// size.
			if projector == nil || e.size() > projector.size() {
				projector = &entries[i]
			}
			continue
		}
		tag := parsed.Tag
		if tag == "" {
			tag = "latest"
		}
		g, ok := groups[tag]
		if !ok {
			g = &group{tag: tag}
			groups[tag] = g
		}
		g.parts = append(g.parts, entries[i])
	}

	chosen, err := pick(ref, groups)
	if err != nil {
		return nil, err
	}
	sort.Slice(chosen.parts, func(i, j int) bool { return chosen.parts[i].Path < chosen.parts[j].Path })

	plan := &Plan{
		Repo:      ref.Repo,
		Revision:  ref.Revision,
		Name:      modelName(ref.Repo, chosen.parts[0].Path),
		Tag:       chosen.tag,
		Weights:   chosen.parts,
		Projector: projector,
	}
	for _, e := range chosen.parts {
		plan.Size += e.size()
	}
	return plan, nil
}

func pick(ref Ref, groups map[string]*group) (*group, error) {
	if ref.File != "" {
		parsed := store.ParseGGUFName(ref.File)
		tag := parsed.Tag
		if tag == "" {
			tag = "latest"
		}
		g, ok := groups[tag]
		if !ok {
			return nil, fmt.Errorf("%s is not in %s@%s", ref.File, ref.Repo, ref.Revision)
		}
		return g, nil
	}

	tags := make([]string, 0, len(groups))
	for tag := range groups {
		tags = append(tags, tag)
	}
	sort.Strings(tags)

	if ref.Quant == "" {
		if len(tags) == 1 {
			return groups[tags[0]], nil
		}
		return nil, fmt.Errorf("%s holds %d quants, name one: %s",
			ref.Repo, len(tags), strings.Join(tags, ", "))
	}
	if g, ok := groups[ref.Quant]; ok {
		return g, nil
	}
	return nil, fmt.Errorf("%s has no %s, only: %s", ref.Repo, ref.Quant, strings.Join(tags, ", "))
}

// modelName prefers what the filename says the model is, and falls back to the
// repo, whose -GGUF suffix names the format rather than the model.
func modelName(repo, file string) string {
	if name := store.ParseGGUFName(file).Name; name != "" {
		return name
	}
	_, base, _ := strings.Cut(repo, "/")
	base = strings.TrimSuffix(base, "-GGUF")
	base = strings.TrimSuffix(base, "-gguf")
	return store.Slug(base)
}

// FileURL is where one repo file is downloaded from.
func (c *Client) FileURL(repo, revision, path string) string {
	return fmt.Sprintf("%s/%s/resolve/%s/%s", c.Endpoint, repo, revision, path)
}

func (c *Client) auth(req *http.Request) {
	if c.Token != "" {
		req.Header.Set("Authorization", "Bearer "+c.Token)
	}
}
