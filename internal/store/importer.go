package store

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"syscall"
)

type ImportLink struct {
	From string
	To   string
}

// Skip says why there is nothing to do, and is empty when Links has work.
type ImportAction struct {
	Model string
	Name  string
	Links []ImportLink
	Skip  string
}

// Nothing is written, so the caller can read the plan first: linking a hundred
// and fifty gigabytes of blobs is worth reviewing.
func PlanImport(src Source, destRoot string) ([]ImportAction, error) {
	models, err := src.List()
	if err != nil {
		return nil, err
	}

	out := make([]ImportAction, 0, len(models))
	for i := range models {
		out = append(out, planOne(&models[i], destRoot))
	}
	return out, nil
}

func planOne(m *Model, destRoot string) ImportAction {
	repo, tag := splitName(m.Name)
	if tag == "" {
		tag = "latest"
	}
	tag = importTag(tag)
	name := slugPath(repo)
	act := ImportAction{Model: m.Name, Name: name + ":" + tag}

	rel, ok := safeRel(name)
	if !ok {
		act.Skip = "name does not map onto a directory"
		return act
	}
	dir := filepath.Join(destRoot, rel)

	want := []ImportLink{{From: m.ModelPath, To: filepath.Join(dir, tag+".gguf")}}
	if m.ProjectorPath != "" {
		want = append(want, ImportLink{
			From: m.ProjectorPath,
			To:   filepath.Join(dir, tag+".mmproj.gguf"),
		})
	}

	for _, l := range want {
		switch linked, err := sameFile(l.From, l.To); {
		case err != nil:
			act.Skip = err.Error()
			return act
		case linked:
			continue
		default:
			act.Links = append(act.Links, l)
		}
	}
	if len(act.Links) == 0 {
		act.Skip = "already linked"
	}
	return act
}

// A different file under the same name is somebody else's model and the blob it
// would replace cannot be recovered, so it is an error.
func sameFile(from, to string) (bool, error) {
	dst, err := os.Stat(to)
	if os.IsNotExist(err) {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	src, err := os.Stat(from)
	if err != nil {
		return false, err
	}
	if !os.SameFile(src, dst) {
		return false, fmt.Errorf("%s already exists and is a different file", to)
	}
	return true, nil
}

// Hardlinks are free and reversible, unlike a copy of a hundred gigabytes, so a
// cross-device link fails rather than falling back to one.
func (a ImportAction) Link() error {
	for i, l := range a.Links {
		if err := os.MkdirAll(filepath.Dir(l.To), 0o755); err != nil {
			return err
		}
		if err := os.Link(l.From, l.To); err != nil {
			// A model half in the store would be served as one, so undo the links already made.
			for _, done := range a.Links[:i] {
				os.Remove(done.To)
			}
			return fmt.Errorf("%s: %w", a.Model, explainLink(err, l))
		}
	}
	return nil
}

func explainLink(err error, l ImportLink) error {
	switch {
	case errors.Is(err, syscall.EXDEV):
		return fmt.Errorf("%s and %s are on different filesystems, so there is no link to make",
			filepath.Dir(l.From), filepath.Dir(l.To))
	case errors.Is(err, os.ErrPermission):
		return fmt.Errorf("%s is owned by another user and fs.protected_hardlinks forbids "+
			"linking it; run the import as that user or as root", l.From)
	}
	return err
}

// Drops the UD- marker so the same weights get the same tag from a pull or an
// import. Non-quant tags such as "latest" and "9b" are left alone.
func importTag(tag string) string {
	tag = slug(tag)
	rest, ok := strings.CutPrefix(tag, "ud-")
	if !ok {
		return tag
	}
	if head, _, _ := strings.Cut(rest, "-"); isQuantToken(head) {
		return rest
	}
	return tag
}

func slugPath(name string) string {
	parts := strings.Split(name, "/")
	for i, p := range parts {
		parts[i] = slug(p)
	}
	return strings.Join(parts, "/")
}
