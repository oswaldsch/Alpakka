// Package supervisor owns the llama-server child process, at most one at a time.
package supervisor

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"os/exec"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/oswaldsch/alpakka/internal/config"
	"github.com/oswaldsch/alpakka/internal/wol"
)

type Logf func(format string, args ...any)

type Supervisor struct {
	llama config.Llama
	// An endpoint absent here is assumed already running, as before Wake-on-LAN existed.
	wol  map[string]string
	logf Logf

	// Thirteen gigabytes off a cold page cache takes tens of seconds, so this is generous.
	LoadTimeout time.Duration

	mu  sync.Mutex
	cur *Instance
	// Kept after cur is torn down, so the log of a failed or evicted load can still be read.
	last *Instance
}

var ErrNotLoaded = errors.New("not loaded")

func New(llama config.Llama, wolMACs map[string]string, logf Logf) *Supervisor {
	if logf == nil {
		logf = func(string, ...any) {}
	}
	return &Supervisor{llama: llama, wol: wolMACs, logf: logf, LoadTimeout: 5 * time.Minute}
}

// Concurrent callers wanting the same runtime share one load. The returned
// instance holds a reference the caller must Release.
func (s *Supervisor) Ensure(ctx context.Context, rt config.Runtime) (*Instance, error) {
	for {
		s.mu.Lock()
		cur := s.cur

		if cur != nil && cur.rt == rt && !cur.dead() {
			s.mu.Unlock()
			if err := cur.wait(ctx); err != nil {
				return nil, err
			}
			if cur.acquireIfLive() {
				return cur, nil
			}
			// It exited between becoming ready and being claimed, so start over.
			continue
		}

		if cur != nil {
			s.mu.Unlock()
			// Reload rather than serve with the previous process's settings, but not while
			// it is streaming, since killing it truncates the answer with no error.
			if cur.Busy() {
				s.logf("%s is busy; waiting for it to finish before reloading", cur.rt.Model)
			}
			if err := cur.waitIdle(ctx); err != nil {
				return nil, err
			}
			s.mu.Lock()
			if s.cur == cur {
				s.logf("reloading: %s", describeChange(cur.rt, rt))
				cur.stop()
				s.cur = nil
			}
			s.mu.Unlock()
			continue
		}

		inst, err := s.start(rt)
		if err != nil {
			s.mu.Unlock()
			return nil, err
		}
		s.cur = inst
		s.last = inst
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
		if inst.acquireIfLive() {
			return inst, nil
		}
		return nil, fmt.Errorf("llama-server exited immediately after loading %s:\n%s",
			rt.Model, inst.diagnosis())
	}
}

// Returns nil rather than loading when the resident instance serves a different model or mode,
// so the caller can fall back to Ensure.
func (s *Supervisor) AcquireIfModel(ctx context.Context, model string, embedding bool) *Instance {
	cur := s.Current()
	if cur == nil || cur.rt.Model != model || cur.rt.Embedding != embedding {
		return nil
	}
	if err := cur.wait(ctx); err != nil || !cur.acquireIfLive() {
		return nil
	}
	return cur
}

func (s *Supervisor) Current() *Instance {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.cur != nil && s.cur.dead() {
		return nil
	}
	return s.cur
}

func (s *Supervisor) Stop() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.cur != nil {
		s.cur.stop()
		s.cur = nil
	}
}

// Last is the most recently started instance, whether or not it is still running.
func (s *Supervisor) Last() *Instance {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.last
}

// Unload stops the resident process ahead of its keep-alive and returns the model it
// served. An empty model means whatever is resident, and nothing resident is not an
// error then. Like a reload, it waits for responses still streaming rather than cut them
// off, unless force is set.
func (s *Supervisor) Unload(ctx context.Context, model string, force bool) (string, error) {
	for {
		cur := s.Current()
		if cur == nil || (model != "" && cur.rt.Model != model) {
			if model == "" {
				return "", nil
			}
			return "", fmt.Errorf("%s: %w", model, ErrNotLoaded)
		}
		if !force {
			// A load still in progress has a caller waiting on it, which gets its answer first.
			_ = cur.wait(ctx)
			if err := cur.waitIdle(ctx); err != nil {
				return "", err
			}
		}

		s.mu.Lock()
		if s.cur == cur && (force || !cur.Busy()) {
			how := "requested"
			if cur.Busy() {
				how = "forced, cutting off a response in progress"
			}
			s.logf("unloading %s: %s", cur.rt.Model, how)
			cur.stop()
			s.cur = nil
			s.mu.Unlock()
			return cur.rt.Model, nil
		}
		// Replaced, or picked up another request, since it went idle.
		s.mu.Unlock()
	}
}

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

	if rt.RPCServers != "" {
		s.wakeRPCNodes(rt.RPCServers)
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
	inst.idle = sync.NewCond(&inst.mu)

	args := Args(rt, port)
	// No shell: a wrapper process would absorb the kill and leave the real server
	// holding the port and the card.
	cmd := exec.Command(llama.Binary(), args...)
	// ggml resolves libggml-hip.so from the executable and working directories,
	// otherwise the server silently runs on CPU.
	cmd.Dir = llama.BackendDir()
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}

	cmd.Stderr = &stderrWriter{inst: inst}
	cmd.Stdout = nil

	s.logf("starting %s on :%d (%s)", rt.Model, port, llama.BackendDir())
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("starting llama-server: %w", err)
	}
	inst.cmd = cmd

	if rt.RPCServers != "" {
		// The RPC node stays reachable only while this machine's network does, so a
		// suspend would silently stall its requests.
		inh, err := startInhibitor(rt.Model)
		if err != nil {
			s.logf("starting sleep inhibitor for %s: %v", rt.Model, err)
		} else {
			inst.inhibit = inh
		}
	}

	go inst.reap()
	go inst.probe(s.LoadTimeout, s.logf)

	return inst, nil
}

// Does not wait for a node to boot, since llama-server's connection failure
// already explains that.
func (s *Supervisor) wakeRPCNodes(addrs string) {
	for _, addr := range strings.Split(addrs, ",") {
		mac, ok := s.wol[addr]
		if !ok {
			continue
		}
		if err := wol.Wake(mac); err != nil {
			s.logf("waking RPC node %s (%s): %v", addr, mac, err)
			continue
		}
		s.logf("sent Wake-on-LAN to %s (%s)", addr, mac)
	}
}

// The gap before llama-server binds is small, and a collision fails the start immediately.
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
	if old.CacheTypeKDraft != new.CacheTypeKDraft || old.CacheTypeVDraft != new.CacheTypeVDraft {
		diffs = append(diffs, fmt.Sprintf("draft cache %s/%s -> %s/%s",
			old.CacheTypeKDraft, old.CacheTypeVDraft, new.CacheTypeKDraft, new.CacheTypeVDraft))
	}
	if old.NumBatch != new.NumBatch || old.NumUBatch != new.NumUBatch {
		diffs = append(diffs, fmt.Sprintf("batch %d/%d -> %d/%d",
			old.NumBatch, old.NumUBatch, new.NumBatch, new.NumUBatch))
	}
	if old.LoadMode != new.LoadMode {
		diffs = append(diffs, fmt.Sprintf("load_mode %q -> %q", old.LoadMode, new.LoadMode))
	}
	if old.SpecType != new.SpecType || old.SpecDraftNMax != new.SpecDraftNMax {
		diffs = append(diffs, fmt.Sprintf("spec %s n=%d -> %s n=%d",
			old.SpecType, old.SpecDraftNMax, new.SpecType, new.SpecDraftNMax))
	}
	if old.NumCPUMoE != new.NumCPUMoE {
		diffs = append(diffs, fmt.Sprintf("num_cpu_moe %d -> %d", old.NumCPUMoE, new.NumCPUMoE))
	}
	if old.OverrideTensor != new.OverrideTensor {
		diffs = append(diffs, fmt.Sprintf("override_tensor %q -> %q",
			old.OverrideTensor, new.OverrideTensor))
	}
	if old.RPCServers != new.RPCServers {
		diffs = append(diffs, fmt.Sprintf("rpc_servers %q -> %q",
			old.RPCServers, new.RPCServers))
	}
	if old.Device != new.Device || old.TensorSplit != new.TensorSplit ||
		old.SplitMode != new.SplitMode || old.MainGPU != new.MainGPU ||
		old.NoKVOffload != new.NoKVOffload {
		diffs = append(diffs, fmt.Sprintf("placement device=%q split=%q mode=%q main_gpu=%d no_kv_offload=%t -> device=%q split=%q mode=%q main_gpu=%d no_kv_offload=%t",
			old.Device, old.TensorSplit, old.SplitMode, old.MainGPU, old.NoKVOffload,
			new.Device, new.TensorSplit, new.SplitMode, new.MainGPU, new.NoKVOffload))
	}
	if old.MoEExpertCache != new.MoEExpertCache ||
		old.MoEExpertCacheInserts != new.MoEExpertCacheInserts {
		diffs = append(diffs, fmt.Sprintf("moe_expert_cache %d/%d -> %d/%d",
			old.MoEExpertCache, old.MoEExpertCacheInserts,
			new.MoEExpertCache, new.MoEExpertCacheInserts))
	}
	if old.KVStreamArenaMiB != new.KVStreamArenaMiB {
		diffs = append(diffs, fmt.Sprintf("kv_stream_arena %d MiB -> %d MiB",
			old.KVStreamArenaMiB, new.KVStreamArenaMiB))
	}
	if old.ProjectorPath != new.ProjectorPath {
		diffs = append(diffs, "projector changed")
	}
	if old.Embedding != new.Embedding {
		diffs = append(diffs, fmt.Sprintf("embeddings %t -> %t", old.Embedding, new.Embedding))
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

// Readiness probes only, generation uses its own client with no timeout.
var httpClient = &http.Client{Timeout: 2 * time.Second}
