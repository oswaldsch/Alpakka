package hub

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"time"
)

var ggufMagic = []byte("GGUF")

// The bytes land in dest+".part" until the whole file is there and its magic checks
// out. Keeping the partial is the point, so a 10 GB pull over a flaky link can resume.
func (c *Client) Download(ctx context.Context, url, dest string, size int64) error {
	if err := os.MkdirAll(filepath.Dir(dest), 0o755); err != nil {
		return err
	}
	part := dest + ".part"

	have := int64(0)
	if info, err := os.Stat(part); err == nil {
		have = info.Size()
	}
	if size > 0 && have > size {
		// The remote file changed under a half-finished download.
		if err := os.Remove(part); err != nil {
			return err
		}
		have = 0
	}

	if size == 0 || have < size {
		var err error
		have, err = c.fetch(ctx, url, part, have, size)
		if err != nil {
			return err
		}
	}

	if size > 0 && have != size {
		os.Remove(part)
		return fmt.Errorf("%s: got %d bytes of %d", filepath.Base(dest), have, size)
	}
	if err := checkMagic(part); err != nil {
		os.Remove(part)
		return err
	}
	return os.Rename(part, dest)
}

func (c *Client) fetch(ctx context.Context, url, part string, have, size int64) (int64, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return have, err
	}
	c.auth(req)
	if have > 0 {
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-", have))
	}

	resp, err := c.HTTP.Do(req)
	if err != nil {
		return have, err
	}
	defer resp.Body.Close()

	flags := os.O_CREATE | os.O_WRONLY
	switch resp.StatusCode {
	case http.StatusOK:
		// The server ignored the range, so what is on disk is not a prefix of what is arriving.
		flags |= os.O_TRUNC
		have = 0
	case http.StatusPartialContent:
		flags |= os.O_APPEND
	case http.StatusRequestedRangeNotSatisfiable:
		return have, nil
	default:
		return have, fmt.Errorf("%s: %s", url, resp.Status)
	}

	if size == 0 && resp.ContentLength > 0 {
		size = have + resp.ContentLength
	}

	f, err := os.OpenFile(part, flags, 0o644)
	if err != nil {
		return have, err
	}
	defer f.Close()

	n, err := io.Copy(f, c.progress(filepath.Base(part), have, size, resp.Body))
	have += n
	if err != nil {
		return have, err
	}
	return have, f.Sync()
}

func (c *Client) progress(name string, done, total int64, r io.Reader) io.Reader {
	const every = 15 * time.Second
	start := time.Now()
	last := start
	return readerFunc(func(p []byte) (int, error) {
		n, err := r.Read(p)
		done += int64(n)
		if time.Since(last) >= every {
			last = time.Now()
			rate := float64(done) / time.Since(start).Seconds() / (1 << 20)
			if total > 0 {
				c.Logf("pull %s: %d%% of %s at %.0f MiB/s",
					name, done*100/total, human(total), rate)
			} else {
				c.Logf("pull %s: %s at %.0f MiB/s", name, human(done), rate)
			}
		}
		return n, err
	})
}

type readerFunc func([]byte) (int, error)

func (f readerFunc) Read(p []byte) (int, error) { return f(p) }

// An HTML error page saved under a .gguf name is what this catches.
func checkMagic(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	head := make([]byte, len(ggufMagic))
	if _, err := io.ReadFull(f, head); err != nil {
		if errors.Is(err, io.ErrUnexpectedEOF) || errors.Is(err, io.EOF) {
			return fmt.Errorf("%s: truncated", filepath.Base(path))
		}
		return err
	}
	if string(head) != string(ggufMagic) {
		return fmt.Errorf("%s: not a GGUF file", filepath.Base(path))
	}
	return nil
}

func human(n int64) string {
	switch {
	case n >= 1<<30:
		return fmt.Sprintf("%.1f GiB", float64(n)/(1<<30))
	case n >= 1<<20:
		return fmt.Sprintf("%.0f MiB", float64(n)/(1<<20))
	default:
		return fmt.Sprintf("%d B", n)
	}
}
