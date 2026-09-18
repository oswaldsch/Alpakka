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
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/oswald/alpakka/internal/config"
	"github.com/oswald/alpakka/internal/wol"
)

// Logf receives human-readable supervisor events.
type Logf func(format string, args ...any)

// Supervisor manages the single llama-server process.
type Supervisor struct {
	llama config.Llama
	// wol maps an RPC endpoint's "host:port" to the MAC address behind it, from
	// Config.WoL. An endpoint absent here is assumed already running, as every
	// endpoint was before Wake-on-LAN existed.
	wol  map[string]string
	logf Logf

	// LoadTimeout bounds a model load. Thirteen gigabytes off a cold page
	// cache takes tens of seconds, so this is deliberately generous.
	LoadTimeout time.Duration

	mu  sync.Mutex
	cur *Instance
}

// New creates a supervisor driving the given llama.cpp build.
func New(llama config.Llama, wolMACs map[string]string, logf Logf) *Supervisor {
	if logf == nil {
		logf = func(string, ...any) {}
	}
	return &Supervisor{llama: llama, wol: wolMACs, logf: logf, LoadTimeout: 5 * time.Minute}
}

// Ensure returns a ready instance running exactly rt, starting or replacing the
// current process if it does not already match.
//
// Concurrent callers that want the same runtime share one load: the second
// caller waits on the same readiness signal instead of starting a rival server.
//
// The returned instance carries a reference held on the caller's behalf, so
// neither the evictor nor a competing reload can stop it. The caller must
// Release it when the request is done.
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
			// It exited between becoming ready and being claimed; start over.
			continue
		}

		if cur != nil {
			s.mu.Unlock()
			// A process-level flag changed, or a different model was asked for.
			// Reload rather than silently serving the request with the settings
			// the previous process happened to have — but not while that
			// process is still streaming a response, because killing it there
			// truncates the answer with no error on either side.
			if cur.busy() {
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

	// A plain writer rather than StderrPipe: os/exec closes a pipe as soon as
	// Wait returns, which races the reader and can swallow the very lines that
	// explain a fast crash. With a writer, Wait waits for the copy to drain.
	cmd.Stderr = &stderrWriter{inst: inst}
	cmd.Stdout = nil

	s.logf("starting %s on :%d (%s)", rt.Model, port, llama.BackendDir())
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("starting llama-server: %w", err)
	}
	inst.cmd = cmd

	if rt.RPCServers != "" {
		// The RPC node this depends on stays reachable only as long as this
		// machine's network does, so a suspend here would silently stall every
		// request the node is carrying.
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

// wakeRPCNodes sends a Wake-on-LAN magic packet to every endpoint in addrs (a
// comma-joined rpc_servers list) that has a MAC configured. It does not wait
// for a node to come up: llama-server's own connection failure already
// explains a load that hits an endpoint still booting.
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

var errStopped = errors.New("llama-server stopped")

// httpClient is used for readiness probes only; generation uses its own client
// with no timeout.
var httpClient = &http.Client{Timeout: 2 * time.Second}
