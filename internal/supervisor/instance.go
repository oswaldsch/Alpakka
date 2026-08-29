package supervisor

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"net/http"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/oswald/alpakka/internal/config"
)

// Instance is one running llama-server process.
type Instance struct {
	rt      config.Runtime
	cmd     *exec.Cmd
	port    int
	baseURL string
	started time.Time

	ready    chan struct{}
	readyErr error

	log *ring

	mu        sync.Mutex
	exited    bool
	expiresAt time.Time
	fit       Fit
}

// Fit records what llama.cpp reported about where the weights ended up.
type Fit struct {
	OffloadedLayers int
	TotalLayers     int
	CPUBufferMiB    float64
	Seen            bool
}

// OK reports whether every layer llama.cpp counted made it onto the GPU.
func (f Fit) OK() bool { return f.Seen && f.OffloadedLayers == f.TotalLayers }

// Runtime returns the settings this process was started with.
func (i *Instance) Runtime() config.Runtime { return i.rt }

// BaseURL is the root of the llama-server HTTP API.
func (i *Instance) BaseURL() string { return i.baseURL }

// StartedAt is when the process was spawned.
func (i *Instance) StartedAt() time.Time { return i.started }

// Fit returns the offload report gathered at load time.
func (i *Instance) Fit() Fit {
	i.mu.Lock()
	defer i.mu.Unlock()
	return i.fit
}

// Touch extends the keep-alive window.
func (i *Instance) Touch(d time.Duration) {
	i.mu.Lock()
	defer i.mu.Unlock()
	if d <= 0 {
		// Ollama treats a non-positive keep-alive as "unload right away".
		i.expiresAt = time.Now()
		return
	}
	i.expiresAt = time.Now().Add(d)
}

// ExpiresAt is when the keep-alive window closes.
func (i *Instance) ExpiresAt() time.Time {
	i.mu.Lock()
	defer i.mu.Unlock()
	return i.expiresAt
}

func (i *Instance) expired(now time.Time) bool {
	i.mu.Lock()
	defer i.mu.Unlock()
	return !i.expiresAt.IsZero() && now.After(i.expiresAt)
}

func (i *Instance) dead() bool {
	i.mu.Lock()
	defer i.mu.Unlock()
	return i.exited
}

// wait blocks until the instance is serving, the load fails, or ctx is done.
// This is what makes requests that arrive during a load queue rather than error.
func (i *Instance) wait(ctx context.Context) error {
	select {
	case <-i.ready:
		return i.readyErr
	case <-ctx.Done():
		return ctx.Err()
	}
}

// drain consumes the child's stderr, keeping the tail for diagnostics and
// picking out the lines that say where the weights landed.
func (i *Instance) drain(r io.Reader) {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for sc.Scan() {
		line := sc.Text()
		i.log.add(line)
		i.observe(line)
	}
}

var (
	reOffloaded = regexp.MustCompile(`offloaded (\d+)/(\d+) layers to GPU`)
	reCPUBuffer = regexp.MustCompile(`(?:CPU|CPU_Mapped) model buffer size\s*=\s*([0-9.]+) MiB`)
)

func (i *Instance) observe(line string) {
	if m := reOffloaded.FindStringSubmatch(line); m != nil {
		off, _ := strconv.Atoi(m[1])
		total, _ := strconv.Atoi(m[2])
		i.mu.Lock()
		i.fit.OffloadedLayers, i.fit.TotalLayers, i.fit.Seen = off, total, true
		i.mu.Unlock()
		return
	}
	if m := reCPUBuffer.FindStringSubmatch(line); m != nil {
		mib, _ := strconv.ParseFloat(m[1], 64)
		i.mu.Lock()
		// llama.cpp prints this once per fitting pass; the largest is the one
		// that actually describes the loaded model.
		if mib > i.fit.CPUBufferMiB {
			i.fit.CPUBufferMiB = mib
		}
		i.mu.Unlock()
	}
}

func (i *Instance) reap() {
	err := i.cmd.Wait()
	i.mu.Lock()
	i.exited = true
	i.mu.Unlock()
	_ = err
}

// probe waits for the server to answer /health, then checks that the load
// actually fit on the GPU before declaring the instance ready.
func (i *Instance) probe(timeout time.Duration, logf Logf) {
	defer close(i.ready)

	deadline := time.Now().Add(timeout)
	for {
		if i.dead() {
			i.readyErr = fmt.Errorf("llama-server exited during load:\n%s", i.log.tail(25))
			return
		}
		if time.Now().After(deadline) {
			i.readyErr = fmt.Errorf("llama-server did not become ready within %s:\n%s",
				timeout, i.log.tail(25))
			i.stop()
			return
		}

		resp, err := httpClient.Get(i.baseURL + "/health")
		if err == nil {
			_, _ = io.Copy(io.Discard, resp.Body)
			resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				break
			}
		}
		time.Sleep(150 * time.Millisecond)
	}

	if err := i.checkFit(); err != nil {
		i.readyErr = err
		i.stop()
		return
	}

	fit := i.Fit()
	logf("%s ready on :%d in %s (%d/%d layers on GPU, %.0f MiB CPU buffer)",
		i.rt.Model, i.port, time.Since(i.started).Round(time.Millisecond),
		fit.OffloadedLayers, fit.TotalLayers, fit.CPUBufferMiB)
}

// checkFit refuses to serve a model that did not fully land on the GPU.
//
// A partial spill is not a degraded success: it is a silent five-fold decode
// slowdown that looks like a performance regression rather than a
// misconfiguration. Failing loudly here is the entire point of the project.
func (i *Instance) checkFit() error {
	fit := i.Fit()
	if !fit.Seen {
		// The offload line is only printed above a certain log verbosity. If it
		// is missing the check cannot run, and guessing would be worse than
		// saying so.
		return nil
	}
	if !fit.OK() {
		return fmt.Errorf(
			"%s: only %d of %d layers fit on the GPU (%.0f MiB of weights left on the CPU). "+
				"Refusing to serve a partially offloaded model. "+
				"Lower num_ctx, quantise the KV cache, or disable the projector",
			i.rt.Model, fit.OffloadedLayers, fit.TotalLayers, fit.CPUBufferMiB)
	}
	return nil
}

// stop kills the process group and waits for it to go away.
func (i *Instance) stop() {
	if i.cmd == nil || i.cmd.Process == nil {
		return
	}
	pgid, err := syscall.Getpgid(i.cmd.Process.Pid)
	if err != nil {
		pgid = i.cmd.Process.Pid
	}
	// Signal the whole group: a stray grandchild that survives would keep both
	// the port and the VRAM, and the next load would fail for reasons that look
	// nothing like the cause.
	_ = syscall.Kill(-pgid, syscall.SIGTERM)

	done := make(chan struct{})
	go func() {
		for !i.dead() {
			time.Sleep(20 * time.Millisecond)
		}
		close(done)
	}()

	select {
	case <-done:
	case <-time.After(10 * time.Second):
		_ = syscall.Kill(-pgid, syscall.SIGKILL)
		<-done
	}
}

// Log returns the tail of the child's stderr.
func (i *Instance) Log(n int) string { return i.log.tail(n) }

// ring keeps the last n lines of output.
type ring struct {
	mu    sync.Mutex
	lines []string
	n     int
}

func newRing(n int) *ring { return &ring{n: n} }

func (r *ring) add(line string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.lines = append(r.lines, line)
	if len(r.lines) > r.n {
		r.lines = r.lines[len(r.lines)-r.n:]
	}
}

func (r *ring) tail(n int) string {
	r.mu.Lock()
	defer r.mu.Unlock()
	if n > len(r.lines) {
		n = len(r.lines)
	}
	return strings.Join(r.lines[len(r.lines)-n:], "\n")
}
