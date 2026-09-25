package supervisor

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/oswaldsch/alpakka/internal/config"
)

type Instance struct {
	rt      config.Runtime
	cmd     *exec.Cmd
	port    int
	baseURL string
	started time.Time

	inhibit *exec.Cmd

	ready    chan struct{}
	readyErr error

	log     *ring
	notable *ring

	mu        sync.Mutex
	idle      *sync.Cond
	inflight  int
	keep      time.Duration
	exited    bool
	expiresAt time.Time
	fit       Fit
	kvBuf     map[string]float64
}

type Fit struct {
	OffloadedLayers int
	TotalLayers     int
	CPUBufferMiB    float64
	// The KV cache is resident VRAM that the model's file size does not account for.
	KVBufferMiB float64
	VRAMMiB     float64
	Seen        bool
}

func (f Fit) OK() bool { return f.Seen && f.OffloadedLayers == f.TotalLayers }

func (i *Instance) Runtime() config.Runtime { return i.rt }

func (i *Instance) BaseURL() string { return i.baseURL }

func (i *Instance) StartedAt() time.Time { return i.started }

func (i *Instance) Running() bool { return !i.dead() }

func (i *Instance) Fit() Fit {
	i.mu.Lock()
	defer i.mu.Unlock()
	return i.fit
}

func (i *Instance) Touch(d time.Duration) {
	i.mu.Lock()
	defer i.mu.Unlock()
	i.keep = d
	i.restartKeepAlive()
}

func (i *Instance) restartKeepAlive() {
	if i.keep <= 0 {
		// Ollama treats a non-positive keep-alive as unload when done.
		i.expiresAt = time.Now()
		return
	}
	i.expiresAt = time.Now().Add(i.keep)
}

// A held reference stops the evictor and a competing reload from tearing the
// process down under a streaming response.
func (i *Instance) acquireIfLive() bool {
	i.mu.Lock()
	defer i.mu.Unlock()
	if i.exited {
		return false
	}
	i.inflight++
	i.restartKeepAlive()
	return true
}

// The window restarts here rather than at arrival, as ollama does, so a
// generation longer than keep_alive is not evicted.
func (i *Instance) Release() {
	i.mu.Lock()
	defer i.mu.Unlock()
	if i.inflight > 0 {
		i.inflight--
	}
	i.restartKeepAlive()
	if i.inflight == 0 {
		i.idle.Broadcast()
	}
}

func (i *Instance) Busy() bool {
	i.mu.Lock()
	defer i.mu.Unlock()
	return i.inflight > 0
}

func (i *Instance) waitIdle(ctx context.Context) error {
	stop := context.AfterFunc(ctx, func() {
		i.mu.Lock()
		defer i.mu.Unlock()
		i.idle.Broadcast()
	})
	defer stop()

	i.mu.Lock()
	defer i.mu.Unlock()
	for i.inflight > 0 && !i.exited {
		if err := ctx.Err(); err != nil {
			return err
		}
		i.idle.Wait()
	}
	return nil
}

func (i *Instance) ExpiresAt() time.Time {
	i.mu.Lock()
	defer i.mu.Unlock()
	return i.expiresAt
}

func (i *Instance) expired(now time.Time) bool {
	i.mu.Lock()
	defer i.mu.Unlock()
	// A request in flight pins the process, since evicting would truncate the
	// generation with no error.
	if i.inflight > 0 {
		return false
	}
	return !i.expiresAt.IsZero() && now.After(i.expiresAt)
}

func (i *Instance) dead() bool {
	i.mu.Lock()
	defer i.mu.Unlock()
	return i.exited
}

func (i *Instance) wait(ctx context.Context) error {
	select {
	case <-i.ready:
		return i.readyErr
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (i *Instance) record(line string) {
	i.log.add(line)
	i.observe(line)
}

// Bounds an unterminated tail so a child writing without newlines cannot grow the buffer forever.
const maxLine = 1 << 20

// An io.Writer rather than a goroutine over StderrPipe: os/exec closes the pipe
// when Wait returns, which can swallow the lines explaining a fast crash.
type stderrWriter struct {
	inst *Instance
	buf  bytes.Buffer
}

func (w *stderrWriter) Write(p []byte) (int, error) {
	w.buf.Write(p)
	for {
		line, err := w.buf.ReadString('\n')
		if err != nil {
			// ReadString drained the buffer, so writing the partial line back restores the tail.
			if len(line) > maxLine {
				w.inst.record(line)
			} else {
				w.buf.WriteString(line)
			}
			return len(p), nil
		}
		w.inst.record(strings.TrimRight(line, "\r\n"))
	}
}

var (
	reOffloaded = regexp.MustCompile(`offloaded (\d+)/(\d+) layers to GPU`)
	reCPUBuffer = regexp.MustCompile(`(?:CPU|CPU_Mapped) model buffer size\s*=\s*([0-9.]+) MiB`)
	// Unchecked against a real --kv-stream-arena-mib build, so KVBufferMiB may not
	// mean here what it does otherwise.
	reKVBuffer = regexp.MustCompile(`(\S+)\s+KV buffer size\s*=\s*([0-9.]+) MiB`)
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
	if isNotable(line) {
		i.notable.add(line)
	}
	if m := reKVBuffer.FindStringSubmatch(line); m != nil {
		// TODO: with no_kv_offload the cache is a "CPU" buffer, which this still
		// counts as KVBufferMiB and /api/ps size_vram although it is host RAM.
		mib, _ := strconv.ParseFloat(m[2], 64)
		i.mu.Lock()
		// Keyed by buffer name at its maximum, so repeated fitting passes count the cache once.
		if i.kvBuf == nil {
			i.kvBuf = map[string]float64{}
		}
		if mib > i.kvBuf[m[1]] {
			i.kvBuf[m[1]] = mib
		}
		total := 0.0
		for _, v := range i.kvBuf {
			total += v
		}
		i.fit.KVBufferMiB = total
		i.mu.Unlock()
		return
	}
	if m := reCPUBuffer.FindStringSubmatch(line); m != nil {
		mib, _ := strconv.ParseFloat(m[1], 64)
		i.mu.Lock()
		// Printed once per fitting pass, and the largest describes the loaded model.
		if mib > i.fit.CPUBufferMiB {
			i.fit.CPUBufferMiB = mib
		}
		i.mu.Unlock()
	}
}

// llama-server prefixes each line with a timestamp and a level letter, like "0.01.563.011 E ...".
var reLevel = regexp.MustCompile(`^[0-9.]+ ([A-Z]) `)

// The default log is dominated by per-layer debug output that would bury the cause of a failed load.
func isNotable(line string) bool {
	if m := reLevel.FindStringSubmatch(line); m != nil {
		switch m[1] {
		case "E", "W":
			return true
		}
	}
	lower := strings.ToLower(line)
	for _, needle := range []string{
		"failed", "error", "cannot", "unable", "insufficient",
		"out of memory", "not enough", "terminate", "abort",
	} {
		if strings.Contains(lower, needle) {
			return true
		}
	}
	return false
}

func (i *Instance) diagnosis() string {
	if s := i.notable.tail(12); s != "" {
		return s
	}
	return i.log.tail(20)
}

func (i *Instance) reap() {
	// Wait also waits for the stderr copier, so every line is filed by the time the process is marked dead.
	err := i.cmd.Wait()
	i.mu.Lock()
	i.exited = true
	// Anything blocked waiting for this instance to go idle never will.
	i.idle.Broadcast()
	i.mu.Unlock()
	_ = err
}

func (i *Instance) probe(timeout time.Duration, logf Logf) {
	defer close(i.ready)

	deadline := time.Now().Add(timeout)
	for {
		if i.dead() {
			i.readyErr = fmt.Errorf("llama-server exited during load:\n%s", i.diagnosis())
			return
		}
		if time.Now().After(deadline) {
			i.readyErr = fmt.Errorf("llama-server did not become ready within %s:\n%s",
				timeout, i.diagnosis())
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
	if err := i.checkVRAMCap(); err != nil {
		i.readyErr = err
		i.stop()
		return
	}

	fit := i.Fit()
	// Weights on the CPU pass the fit check only when put there on purpose, so say how much landed there.
	cpu := ""
	if fit.CPUBufferMiB > 0 {
		cpu = fmt.Sprintf(", %.0f MiB weights on CPU", fit.CPUBufferMiB)
	}
	logf("%s ready on :%d in %s (%d/%d layers on GPU, %.0f MiB process VRAM, %.0f MiB KV cache%s)",
		i.rt.Model, i.port, time.Since(i.started).Round(time.Millisecond),
		fit.OffloadedLayers, fit.TotalLayers, fit.VRAMMiB, fit.KVBufferMiB, cpu)
}

var reDRMVRAM = regexp.MustCompile(`(?m)^drm-memory-vram:\s*([0-9]+)\s*KiB\s*$`)

func parseVRAMKiB(s string) uint64 {
	var total uint64
	for _, m := range reDRMVRAM.FindAllStringSubmatch(s, -1) {
		n, _ := strconv.ParseUint(m[1], 10, 64)
		total += n
	}
	return total
}

// A configured cap fails closed if no allocation can be measured.
func (i *Instance) checkVRAMCap() error {
	if i.rt.GPUVRAMCapMiB <= 0 {
		return nil
	}
	fdinfo := filepath.Join("/proc", strconv.Itoa(i.cmd.Process.Pid), "fdinfo")
	entries, err := os.ReadDir(fdinfo)
	if err != nil {
		return fmt.Errorf("%s: cannot verify %d MiB GPU VRAM cap: %w", i.rt.Model, i.rt.GPUVRAMCapMiB, err)
	}
	var kib uint64
	for _, entry := range entries {
		b, readErr := os.ReadFile(filepath.Join(fdinfo, entry.Name()))
		if readErr == nil {
			kib += parseVRAMKiB(string(b))
		}
	}
	if kib == 0 {
		return fmt.Errorf("%s: kernel DRM fdinfo did not report VRAM; refusing an unverifiable %d MiB cap", i.rt.Model, i.rt.GPUVRAMCapMiB)
	}
	mib := float64(kib) / 1024
	i.mu.Lock()
	i.fit.VRAMMiB = mib
	i.mu.Unlock()
	if mib > float64(i.rt.GPUVRAMCapMiB) {
		return fmt.Errorf("%s: llama-server uses %.0f MiB GPU VRAM, exceeding the configured %d MiB cap", i.rt.Model, mib, i.rt.GPUVRAMCapMiB)
	}
	return nil
}

// A partial spill is a silent five-fold decode slowdown that looks like a
// regression, so fail loudly.
func (i *Instance) checkFit() error {
	fit := i.Fit()
	if !fit.Seen {
		// An unverifiable load fails rather than passes, else the guarantee becomes a
		// no-op the first time llama.cpp rewords the line.
		return fmt.Errorf(
			"%s: llama-server never reported its layer offload, so alpakka cannot tell "+
				"whether the model fits on the GPU and will not serve it. "+
				"This build should log \"offloaded N/M layers to GPU\" at -lv %s; "+
				"if it no longer does, alpakka needs updating for it",
			i.rt.Model, logVerbosity)
	}
	if !fit.OK() && !i.rt.AllowPartialOffload {
		return fmt.Errorf(
			"%s: only %d of %d layers fit on the GPU (%.0f MiB of weights left on the CPU). "+
				"Refusing to serve a partially offloaded model. "+
				"Lower num_ctx, quantise the KV cache, or disable the projector",
			i.rt.Model, fit.OffloadedLayers, fit.TotalLayers, fit.CPUBufferMiB)
	}
	if i.rt.AllowPartialOffload && fit.OffloadedLayers > i.rt.NumGPU {
		return fmt.Errorf(
			"%s: llama-server reported %d GPU layers despite the deliberate num_gpu cap of %d",
			i.rt.Model, fit.OffloadedLayers, i.rt.NumGPU)
	}
	return nil
}

func (i *Instance) stop() {
	if i.cmd != nil && i.cmd.Process != nil {
		pgid, err := syscall.Getpgid(i.cmd.Process.Pid)
		if err != nil {
			pgid = i.cmd.Process.Pid
		}
		// Signal the whole group: a surviving grandchild would keep the port and VRAM
		// and break the next load confusingly.
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
	stopInhibitor(i.inhibit)
}

func (i *Instance) Log(n int) string { return i.log.tail(n) }

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
