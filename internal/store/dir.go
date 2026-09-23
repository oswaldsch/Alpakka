package store

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/oswald/alpakka/internal/gguf"
)

// Layout is <root>/<name>/<tag>.gguf and a name may nest. A sibling <tag>.mmproj.gguf is
// the vision projector, and llama.cpp splits keep their <tag>-00001-of-0000N.gguf naming.
type DirStore struct {
	root  string
	cache *ggufCache
}

func NewDir(root string) *DirStore {
	return &DirStore{root: root, cache: newGGUFCache()}
}

func (s *DirStore) Root() string { return s.root }

type tagFiles struct {
	parts     []string
	projector string
}

func (s *DirStore) List() ([]Model, error) {
	dirs, err := s.scan()
	if err != nil {
		return nil, err
	}

	var models []Model
	for _, name := range sortedKeys(dirs) {
		for _, tag := range sortedKeys(dirs[name]) {
			m, err := s.model(name, tag, dirs[name][tag])
			if err != nil {
				// A GGUF that will not parse is not a model, and dropping it keeps a half-written
				// download out of /api/tags.
				continue
			}
			models = append(models, *m)
		}
	}

	sortNewestFirst(models)
	return models, nil
}

func (s *DirStore) Get(name string) (*Model, error) {
	repo, want := splitName(name)
	rel, ok := safeRel(repo)
	if !ok {
		return nil, fmt.Errorf("%q: %w", name, ErrNotFound)
	}

	tags, err := s.scanDir(filepath.Join(s.root, rel))
	if err != nil || len(tags) == 0 {
		return nil, fmt.Errorf("%q: %w", name, ErrNotFound)
	}

	if want == "" {
		have := sortedKeys(tags)
		if len(have) != 1 {
			return nil, fmt.Errorf("model %q has %d tags, name one of: %s",
				repo, len(have), strings.Join(have, ", "))
		}
		want = have[0]
	}
	files, ok := tags[want]
	if !ok {
		return nil, fmt.Errorf("%q: %w", name, ErrNotFound)
	}
	return s.model(repo, want, files)
}

func (s *DirStore) Tags(name string) []string {
	rel, ok := safeRel(name)
	if !ok {
		return nil
	}
	tags, err := s.scanDir(filepath.Join(s.root, rel))
	if err != nil {
		return nil
	}
	return sortedKeys(tags)
}

func (s *DirStore) Path(name, tag string) (string, bool) {
	rel, ok := safeRel(name)
	if !ok || !safeSegment(tag) {
		return "", false
	}
	return filepath.Join(s.root, rel, tag+".gguf"), true
}

func (s *DirStore) scan() (map[string]map[string]tagFiles, error) {
	out := map[string]map[string]tagFiles{}

	err := filepath.WalkDir(s.root, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			if p == s.root {
				return err
			}
			// An unreadable subdirectory is not a reason to report no models.
			return nil
		}
		if !d.IsDir() || p == s.root {
			return nil
		}
		tags, err := s.scanDir(p)
		if err != nil || len(tags) == 0 {
			return nil
		}
		rel, err := filepath.Rel(s.root, p)
		if err != nil {
			return nil
		}
		out[filepath.ToSlash(rel)] = tags
		return nil
	})
	if err != nil && !os.IsNotExist(err) {
		return nil, err
	}
	return out, nil
}

func (s *DirStore) scanDir(dir string) (map[string]tagFiles, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}

	out := map[string]tagFiles{}
	splits := map[string][]string{}
	loose := "" // a projector named for the repo rather than for one tag
	var looseSize int64

	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".gguf") {
			continue
		}
		base := strings.TrimSuffix(e.Name(), ".gguf")
		full := filepath.Join(dir, e.Name())

		if tag, ok := strings.CutSuffix(base, ".mmproj"); ok {
			f := out[tag]
			f.projector = full
			out[tag] = f
			continue
		}

		parsed := ParseGGUFName(e.Name())
		if parsed.Draft {
			// A draft model is not servable on its own, serving one answers with the draft's output.
			continue
		}
		if parsed.Projector {
			// Publishers ship F32 beside F16, and the larger is better at no cost at this size.
			if info, err := e.Info(); err == nil && (loose == "" || info.Size() > looseSize) {
				loose, looseSize = full, info.Size()
			}
			continue
		}

		stem := base
		if parsed.Parts > 0 {
			stem, _, _ = splitPart(base)
		}
		// A file still under its publisher's name is tagged by its quant, so downloads
		// read without being renamed.
		tag := stem
		if _, taken := out[parsed.Tag]; parsed.Tag != "" && !taken {
			tag = parsed.Tag
		}

		if parsed.Parts > 0 {
			splits[tag] = append(splits[tag], full)
			continue
		}
		f := out[tag]
		f.parts = []string{full}
		out[tag] = f
	}

	for tag, parts := range splits {
		sort.Strings(parts)
		f := out[tag]
		f.parts = parts
		out[tag] = f
	}
	for tag, f := range out {
		if len(f.parts) == 0 {
			delete(out, tag)
			continue
		}
		if f.projector == "" && loose != "" {
			f.projector = loose
			out[tag] = f
		}
	}
	return out, nil
}

func (s *DirStore) model(name, tag string, files tagFiles) (*Model, error) {
	m := &Model{
		Name:          name + ":" + tag,
		ModelPath:     files.parts[0],
		ProjectorPath: files.projector,
		Params:        map[string]any{},
		cache:         s.cache,
	}

	paths := files.parts
	if files.projector != "" {
		paths = append(append([]string{}, paths...), files.projector)
	}
	for _, p := range paths {
		info, err := os.Stat(p)
		if err != nil {
			return nil, err
		}
		m.Size += info.Size()
		if info.ModTime().After(m.ModifiedAt) {
			m.ModifiedAt = info.ModTime()
		}
	}

	f, err := m.GGUF()
	if err != nil {
		return nil, err
	}
	m.Config = ggufConfig(f)
	// llama.cpp reads the template from the GGUF itself. It is carried here only so
	// /api/show can render a Modelfile.
	m.Template, _ = f.String("tokenizer.chat_template")
	m.Digest = headerDigest(m.Name, m.Size, f)
	return m, nil
}

func ggufConfig(f *gguf.File) Config {
	cfg := Config{ModelFormat: "gguf", FileType: "unknown"}
	if arch := f.Architecture(); arch != "" {
		cfg.ModelFamily = arch
		cfg.ModelFamilies = []string{arch}
	}
	cfg.ModelType = gguf.HumanParams(f.ParameterCount())
	if ft, ok := f.Uint("general.file_type"); ok {
		cfg.FileType = gguf.FileTypeName(ft)
	}
	return cfg
}

// Clients key caches on it, so it must survive restarts and change when weights are
// replaced. The metadata header, size and name separate every model without hashing gigabytes.
func headerDigest(name string, size int64, f *gguf.File) string {
	h := sha256.New()
	fmt.Fprintf(h, "%s\x00%d\x00", name, size)
	for _, k := range sortedKeys(f.KV) {
		fmt.Fprintf(h, "%s=%v\x00", k, f.KV[k])
	}
	return hex.EncodeToString(h.Sum(nil))
}

// llama.cpp's split naming, e.g. "iq3-xxs-00002-of-00003".
const (
	partWidth  = 5
	partMarker = "-of-"
	partSuffix = 1 + partWidth + len(partMarker) + partWidth
)

func splitPart(base string) (tag string, part int, ok bool) {
	if len(base) <= partSuffix {
		return "", 0, false
	}
	tail := base[len(base)-partSuffix:]
	if tail[0] != '-' || tail[1+partWidth:1+partWidth+len(partMarker)] != partMarker {
		return "", 0, false
	}
	digits := tail[1:1+partWidth] + tail[1+partWidth+len(partMarker):]
	for _, r := range digits {
		if r < '0' || r > '9' {
			return "", 0, false
		}
	}
	for _, r := range tail[1 : 1+partWidth] {
		part = part*10 + int(r-'0')
	}
	return base[:len(base)-partSuffix], part, true
}

// Leaves a colon in a registry host alone.
func splitName(name string) (string, string) {
	p := path(name)
	i := strings.LastIndex(p, ":")
	if i < 0 {
		return name, ""
	}
	off := len(name) - len(p)
	return name[:off+i], name[off+i+1:]
}

// The name arrives from an HTTP request, so it must not point anywhere else.
func safeRel(name string) (string, bool) {
	if name == "" {
		return "", false
	}
	parts := strings.Split(name, "/")
	for _, part := range parts {
		if !safeSegment(part) {
			return "", false
		}
	}
	return filepath.Join(parts...), true
}

func safeSegment(s string) bool {
	return s != "" && s != "." && s != ".." && !strings.ContainsAny(s, `/\`)
}

func sortNewestFirst(models []Model) {
	sort.Slice(models, func(i, j int) bool {
		return models[i].ModifiedAt.After(models[j].ModifiedAt)
	})
}

func sortedKeys[V any](m map[string]V) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

var _ Source = (*DirStore)(nil)
