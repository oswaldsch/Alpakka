package store

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"syscall"
)

// ImportLink is one blob and the name it gets in the directory layout.
type ImportLink struct {
	From string
	To   string
}

// ImportAction is what importing one ollama model would do. Skip says why
// there is nothing to do, and is empty when Links has work in it.
type ImportAction struct {
	Model string
	Name  string
	Links []ImportLink
	Skip  string
}

// PlanImport works out how each of src's models maps onto destRoot.
//
// Nothing is written: the caller decides whether to act on the plan, because
// linking a hundred and fifty gigabytes of blobs under new names is the kind
// of thing worth reading before running.
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

// sameFile reports whether To already is From. A different file under the same
// name is an error rather than something to overwrite: it is somebody else's
// model, and the blob it would replace cannot be recovered from here.
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

// Link creates the action's hardlinks.
//
// The blobs run to a hundred gigabytes and are already on this filesystem, so
// a link is free and costs nothing to undo. A copy would be neither, which is
// why a cross-device link fails here rather than falling back to one.
func (a ImportAction) Link() error {
	for i, l := range a.Links {
		if err := os.MkdirAll(filepath.Dir(l.To), 0o755); err != nil {
			return err
		}
		if err := os.Link(l.From, l.To); err != nil {
			// A model half in the store would be served as one, so undo the
			// links this action already made.
			for _, done := range a.Links[:i] {
				os.Remove(done.To)
			}
			return fmt.Errorf("%s: %w", a.Model, explainLink(err, l))
		}
	}
	return nil
}

// explainLink names the two failures that are about the setup rather than the
// model. Neither is a reason to fall back to copying: the blobs are a hundred
// and fifty gigabytes, and a copy is neither free nor reversible.
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

// importTag drops the UD- marker from a quant tag, so the same weights land on
// the same tag whether they arrived through a pull or through an import.
// Tags that are not quants, "latest" and "9b" among them, are left alone.
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

// slugPath slugs each segment of a possibly nested model name.
func slugPath(name string) string {
	parts := strings.Split(name, "/")
	for i, p := range parts {
		parts[i] = slug(p)
	}
	return strings.Join(parts, "/")
}
