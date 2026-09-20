package hub

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/oswald/alpakka/internal/store"
)

// ggufFile is a real GGUF header with no keys and no tensors, which is all the
// magic check and the store reader need.
func ggufFile(pad int) []byte {
	var b bytes.Buffer
	b.WriteString("GGUF")
	var raw [8]byte
	binary.LittleEndian.PutUint32(raw[:4], 3)
	b.Write(raw[:4])
	binary.LittleEndian.PutUint64(raw[:], 0) // tensor count
	b.Write(raw[:])
	binary.LittleEndian.PutUint64(raw[:], 0) // metadata count
	b.Write(raw[:])
	b.Write(bytes.Repeat([]byte{'x'}, pad))
	return b.Bytes()
}

// fakeHub serves the two endpoints a pull needs: the repo tree and the files.
type fakeHub struct {
	files  map[string][]byte
	ranges int // requests that carried a Range header
}

func (h *fakeHub) start(t *testing.T) *Client {
	t.Helper()
	mux := http.NewServeMux()
	mux.HandleFunc("/api/models/", func(w http.ResponseWriter, r *http.Request) {
		var entries []Entry
		for path, body := range h.files {
			entries = append(entries, Entry{Type: "file", Path: path, Size: int64(len(body))})
		}
		_ = json.NewEncoder(w).Encode(entries)
	})
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		_, path, ok := strings.Cut(r.URL.Path, "/resolve/main/")
		body, have := h.files[path]
		if !ok || !have {
			http.NotFound(w, r)
			return
		}
		if rng := r.Header.Get("Range"); rng != "" {
			h.ranges++
			var from int64
			fmt.Sscanf(rng, "bytes=%d-", &from)
			if from >= int64(len(body)) {
				w.WriteHeader(http.StatusRequestedRangeNotSatisfiable)
				return
			}
			w.Header().Set("Content-Range",
				fmt.Sprintf("bytes %d-%d/%d", from, len(body)-1, len(body)))
			w.WriteHeader(http.StatusPartialContent)
			w.Write(body[from:])
			return
		}
		w.Write(body)
	})

	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)

	c := NewClient(func(string, ...any) {})
	c.Endpoint = srv.URL
	return c
}

// The slashes in owner/repo are part of the route, not data.
func TestURLsKeepTheirSeparators(t *testing.T) {
	c := NewClient(nil)
	c.Endpoint = "https://huggingface.co"
	got := c.FileURL("unsloth/Qwen3.8-27B-GGUF", "main", "UD-IQ3_XXS/model.gguf")
	want := "https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/UD-IQ3_XXS/model.gguf"
	if got != want {
		t.Errorf("FileURL = %q, want %q", got, want)
	}
}

func TestResolvePicksTheQuant(t *testing.T) {
	h := &fakeHub{files: map[string][]byte{
		"Qwen3.8-27B-UD-IQ3_XXS.gguf": ggufFile(16),
		"Qwen3.8-27B-UD-Q4_K_XL.gguf": ggufFile(32),
		"mmproj-BF16.gguf":            ggufFile(8),
	}}
	c := h.start(t)

	ref, err := ParseRef("unsloth/Qwen3.8-27B-GGUF:UD-IQ3_XXS")
	if err != nil {
		t.Fatal(err)
	}
	plan, err := c.Resolve(context.Background(), ref)
	if err != nil {
		t.Fatal(err)
	}
	if plan.Name != "qwen3.8-27b" || plan.Tag != "iq3-xxs" {
		t.Errorf("plan = %s:%s", plan.Name, plan.Tag)
	}
	if len(plan.Weights) != 1 || plan.Weights[0].Path != "Qwen3.8-27B-UD-IQ3_XXS.gguf" {
		t.Errorf("weights = %+v", plan.Weights)
	}
	if plan.Projector == nil || plan.Projector.Path != "mmproj-BF16.gguf" {
		t.Errorf("projector = %+v", plan.Projector)
	}
}

func TestResolveByFileURL(t *testing.T) {
	h := &fakeHub{files: map[string][]byte{
		"Qwen3.8-27B-UD-IQ3_XXS.gguf": ggufFile(16),
		"Qwen3.8-27B-Q8_0.gguf":       ggufFile(16),
	}}
	c := h.start(t)

	ref, err := ParseRef("https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B-Q8_0.gguf")
	if err != nil {
		t.Fatal(err)
	}
	plan, err := c.Resolve(context.Background(), ref)
	if err != nil {
		t.Fatal(err)
	}
	if plan.Tag != "q8-0" {
		t.Errorf("tag = %q", plan.Tag)
	}
}

func TestResolveNeedsAQuantWhenThereAreSeveral(t *testing.T) {
	h := &fakeHub{files: map[string][]byte{
		"Qwen3.8-27B-UD-IQ3_XXS.gguf": ggufFile(16),
		"Qwen3.8-27B-Q8_0.gguf":       ggufFile(16),
	}}
	c := h.start(t)

	ref, _ := ParseRef("unsloth/Qwen3.8-27B-GGUF")
	_, err := c.Resolve(context.Background(), ref)
	if err == nil {
		t.Fatal("expected an error")
	}
	if !strings.Contains(err.Error(), "iq3-xxs") || !strings.Contains(err.Error(), "q8-0") {
		t.Errorf("error should name the quants, got %v", err)
	}
}

// A repo naming its file only by quant falls back to the repo for the name.
func TestResolveNamesFromTheRepoWhenTheFileCannot(t *testing.T) {
	h := &fakeHub{files: map[string][]byte{"Q4_K_M.gguf": ggufFile(16)}}
	c := h.start(t)

	ref, _ := ParseRef("someone/GLM-4.7-Flash-GGUF")
	plan, err := c.Resolve(context.Background(), ref)
	if err != nil {
		t.Fatal(err)
	}
	if plan.Name != "glm-4.7-flash" || plan.Tag != "q4-k-m" {
		t.Errorf("plan = %s:%s", plan.Name, plan.Tag)
	}
}

func TestPullWritesTheStoreLayout(t *testing.T) {
	h := &fakeHub{files: map[string][]byte{
		"Qwen3.8-27B-UD-IQ3_XXS.gguf": ggufFile(64),
		"mmproj-BF16.gguf":            ggufFile(8),
	}}
	c := h.start(t)
	root := t.TempDir()

	ref, _ := ParseRef("unsloth/Qwen3.8-27B-GGUF:UD-IQ3_XXS")
	plan, err := c.Resolve(context.Background(), ref)
	if err != nil {
		t.Fatal(err)
	}
	if err := c.Pull(context.Background(), plan, root, true, false); err != nil {
		t.Fatal(err)
	}

	for _, want := range []string{"iq3-xxs.gguf", "iq3-xxs.mmproj.gguf"} {
		if _, err := os.Stat(filepath.Join(root, "qwen3.8-27b", want)); err != nil {
			t.Errorf("%s: %v", want, err)
		}
	}
	// A pull must leave nothing half-written behind.
	if _, err := os.Stat(filepath.Join(root, "qwen3.8-27b", "iq3-xxs.gguf.part")); err == nil {
		t.Error("the .part file survived a finished download")
	}

	// What was pulled has to be what the store then serves.
	m, err := store.NewDir(root).Get("qwen3.8-27b:iq3-xxs")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasSuffix(m.ProjectorPath, "iq3-xxs.mmproj.gguf") {
		t.Errorf("ProjectorPath = %q", m.ProjectorPath)
	}
}

func TestPullRefusesToClobberWithoutForce(t *testing.T) {
	h := &fakeHub{files: map[string][]byte{"Qwen3.8-27B-Q8_0.gguf": ggufFile(16)}}
	c := h.start(t)
	root := t.TempDir()

	ref, _ := ParseRef("unsloth/Qwen3.8-27B-GGUF:Q8_0")
	plan, err := c.Resolve(context.Background(), ref)
	if err != nil {
		t.Fatal(err)
	}
	if err := c.Pull(context.Background(), plan, root, false, false); err != nil {
		t.Fatal(err)
	}
	if err := c.Pull(context.Background(), plan, root, false, false); err == nil {
		t.Fatal("a second pull should refuse the existing tag")
	}
	if err := c.Pull(context.Background(), plan, root, false, true); err != nil {
		t.Errorf("-f should replace it: %v", err)
	}
}

func TestPullFetchesEverySplitPart(t *testing.T) {
	h := &fakeHub{files: map[string][]byte{
		"Qwen3.8-27B-UD-IQ3_XXS-00001-of-00003.gguf": ggufFile(16),
		"Qwen3.8-27B-UD-IQ3_XXS-00002-of-00003.gguf": ggufFile(16),
		"Qwen3.8-27B-UD-IQ3_XXS-00003-of-00003.gguf": ggufFile(16),
	}}
	c := h.start(t)
	root := t.TempDir()

	ref, _ := ParseRef("unsloth/Qwen3.8-27B-GGUF:UD-IQ3_XXS")
	plan, err := c.Resolve(context.Background(), ref)
	if err != nil {
		t.Fatal(err)
	}
	if len(plan.Weights) != 3 {
		t.Fatalf("weights = %d, want 3", len(plan.Weights))
	}
	if err := c.Pull(context.Background(), plan, root, false, false); err != nil {
		t.Fatal(err)
	}
	for i := 1; i <= 3; i++ {
		name := fmt.Sprintf("iq3-xxs-%05d-of-00003.gguf", i)
		if _, err := os.Stat(filepath.Join(root, "qwen3.8-27b", name)); err != nil {
			t.Errorf("%s: %v", name, err)
		}
	}
}

// A half-finished download must continue rather than start over.
func TestDownloadResumesFromAPartFile(t *testing.T) {
	body := ggufFile(4096)
	h := &fakeHub{files: map[string][]byte{"model-Q8_0.gguf": body}}
	c := h.start(t)

	dest := filepath.Join(t.TempDir(), "q8-0.gguf")
	if err := os.WriteFile(dest+".part", body[:1000], 0o644); err != nil {
		t.Fatal(err)
	}

	url := c.FileURL("owner/repo", "main", "model-Q8_0.gguf")
	if err := c.Download(context.Background(), url, dest, int64(len(body))); err != nil {
		t.Fatal(err)
	}
	if h.ranges != 1 {
		t.Errorf("sent %d ranged requests, want 1", h.ranges)
	}
	got, err := os.ReadFile(dest)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, body) {
		t.Errorf("resumed file is %d bytes, want %d", len(got), len(body))
	}
}

// A short read is what a dropped connection looks like, and a file that is not
// a GGUF is what a signed-out HTML error page looks like. Neither may be left
// in the store.
func TestDownloadRejectsTruncatedAndNonGGUF(t *testing.T) {
	body := ggufFile(64)
	h := &fakeHub{files: map[string][]byte{
		"model-Q8_0.gguf": body,
		"html-Q4_0.gguf":  []byte("<html>not found</html>"),
	}}
	c := h.start(t)

	short := filepath.Join(t.TempDir(), "short.gguf")
	err := c.Download(context.Background(), c.FileURL("o/r", "main", "model-Q8_0.gguf"),
		short, int64(len(body))+100)
	if err == nil {
		t.Error("a short download should fail")
	}
	if _, err := os.Stat(short + ".part"); err == nil {
		t.Error("a truncated .part file survived")
	}

	html := filepath.Join(t.TempDir(), "html.gguf")
	if err := c.Download(context.Background(), c.FileURL("o/r", "main", "html-Q4_0.gguf"),
		html, 0); err == nil {
		t.Error("a non-GGUF download should fail")
	}
	if _, err := os.Stat(html + ".part"); err == nil {
		t.Error("a non-GGUF .part file survived")
	}
}
