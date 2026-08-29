// Package supervisor owns the llama-server child process: launching it with the
// right flags, deciding when a request needs a different process, and tearing
// the old one down cleanly.
//
// One 16 GB card means one model resident at a time, so there is at most one
// child alive at any moment.
package supervisor

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"os/exec"
	"sync"
	"syscall"
	"time"

	"github.com/oswald/alpakka/internal/config"
)

// Logf receives human-readable supervisor events.
type Logf func(format string, args ...any)

// Supervisor manages the single llama-server process.
type Supervisor struct {
	llama config.Llama
	logf  Logf

	// LoadTimeout bounds a model load. Thirteen gigabytes off a cold page
	// cache takes tens of seconds, so this is deliberately generous.
	LoadTimeout time.Duration

	mu  sync.Mutex
	cur *Instance
}

// New creates a supervisor driving the given llama.cpp build.
func New(llama config.Llama, logf Logf) *Supervisor {
	if logf == nil {
		logf = func(string, ...any) {}
	}
	return &Supervisor{llama: llama, logf: logf, LoadTimeout: 5 * time.Minute}
}

// Ensure returns a ready instance running exactly rt, starting or replacing the
// current process if it does not already match.
//
// Concurrent callers that want the same runtime share one load: the second
// caller waits on the same readiness signal instead of starting a rival server.
func (s *Supervisor) Ensure(ctx context.Context, rt config.Runtime) (*Instance, error) {
	s.mu.Lock()
	if s.cur != nil && s.cur.rt == rt && !s.cur.dead() {
		inst := s.cur
		s.mu.Unlock()
		if err := inst.wait(ctx); err != nil {
			return nil, err
		}
		return inst, nil
	}

	if s.cur != nil {
		// A process-level flag changed. Reload rather than silently serving
		// the request with the settings the previous process happened to have.
		s.logf("reloading: %s", describeChange(s.cur.rt, rt))
		s.cur.stop()
		s.cur = nil
	}

	inst, err := s.start(rt)
	if err != nil {
		s.mu.Unlock()
		return nil, err
	}
	s.cur = inst
	s.mu.Unlock()

	if err := inst.wait(ctx); err != nil {
		s.mu.Lock()
		if s.cur == inst {
			s.cur.stop()
			s.cur = nil
		}
		s.mu.Unlock()
		return nil, err
	}
	return inst, nil
}

// Current returns the running instance, or nil.
func (s *Supervisor) Current() *Instance {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.cur != nil && s.cur.dead() {
		return nil
	}
	return s.cur
}

// Stop tears down the running instance, if any.
func (s *Supervisor) Stop() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.cur != nil {
		s.cur.stop()
		s.cur = nil
	}
}

// EvictExpired stops the running instance if its keep-alive has lapsed.
func (s *Supervisor) EvictExpired(now time.Time) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.cur == nil || !s.cur.expired(now) {
		return false
	}
	s.logf("evicting %s: keep-alive expired", s.cur.rt.Model)
	s.cur.stop()
	s.cur = nil
	return true
}

// RunEvictor evicts expired instances until ctx is cancelled.
func (s *Supervisor) RunEvictor(ctx context.Context, tick time.Duration) {
	t := time.NewTicker(tick)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case now := <-t.C:
			s.EvictExpired(now)
		}
	}
}

func (s *Supervisor) start(rt config.Runtime) (*Instance, error) {
	port, err := freePort()
	if err != nil {
		return nil, fmt.Errorf("finding a port: %w", err)
	}

	llama := s.llama
	if rt.Backend != "" {
		llama.Backend = rt.Backend
	}

	inst := &Instance{
		rt:      rt,
		port:    port,
		baseURL: fmt.Sprintf("http://127.0.0.1:%d", port),
		ready:   make(chan struct{}),
		log:     newRing(400),
		notable: newRing(40),
		started: time.Now(),
	}

	args := Args(rt, port)
	// exec.Command runs the binary directly — no shell, so there is no wrapper
	// process to absorb the eventual kill and leave the real server holding the
	// port and the card.
	cmd := exec.Command(llama.Binary(), args...)
	// ggml resolves libggml-hip.so relative to the executable's directory and
	// the working directory. Without this the server comes up on CPU at a few
	// tokens a second and reports no error at all.
	cmd.Dir = llama.BackendDir()
	// Its own process group, so teardown can signal any grandchildren too.
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}

	stderr, err := cmd.StderrPipe()
	if err != nil {
		return nil, err
	}
	cmd.Stdout = nil

	s.logf("starting %s on :%d (%s)", rt.Model, port, llama.BackendDir())
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("starting llama-server: %w", err)
	}
	inst.cmd = cmd

	go inst.drain(stderr)
	go inst.reap()
	go inst.probe(s.LoadTimeout, s.logf)

	return inst, nil
}

// freePort asks the kernel for an unused port. The gap between closing this
// listener and llama-server binding is small enough to be acceptable, and a
// collision surfaces immediately as a failed start rather than silently.
func freePort() (int, error) {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return 0, err
	}
	defer l.Close()
	return l.Addr().(*net.TCPAddr).Port, nil
}

func describeChange(old, new config.Runtime) string {
	if old.Model != new.Model {
		return fmt.Sprintf("%s -> %s", old.Model, new.Model)
	}
	var diffs []string
	if old.NumCtx != new.NumCtx {
		diffs = append(diffs, fmt.Sprintf("num_ctx %d -> %d", old.NumCtx, new.NumCtx))
	}
	if old.CacheTypeK != new.CacheTypeK || old.CacheTypeV != new.CacheTypeV {
		diffs = append(diffs, fmt.Sprintf("cache %s/%s -> %s/%s",
			old.CacheTypeK, old.CacheTypeV, new.CacheTypeK, new.CacheTypeV))
	}
	if old.SpecType != new.SpecType || old.SpecDraftNMax != new.SpecDraftNMax {
		diffs = append(diffs, fmt.Sprintf("spec %s n=%d -> %s n=%d",
			old.SpecType, old.SpecDraftNMax, new.SpecType, new.SpecDraftNMax))
	}
	if old.ProjectorPath != new.ProjectorPath {
		diffs = append(diffs, "projector changed")
	}
	if len(diffs) == 0 {
		return fmt.Sprintf("%s, settings changed", new.Model)
	}
	out := new.Model + ": "
	for i, d := range diffs {
		if i > 0 {
			out += ", "
		}
		out += d
	}
	return out
}

var errStopped = errors.New("llama-server stopped")

// httpClient is used for readiness probes only; generation uses its own client
// with no timeout.
var httpClient = &http.Client{Timeout: 2 * time.Second}
