package supervisor

import (
	"context"
	"fmt"
	"os"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/oswald/alpakka/internal/config"
	"github.com/oswald/alpakka/internal/store"
)

// Runs against the installed alpakka's own llama.cpp build and model roots,
// or ALPAKKA_LLAMA_LIB_DIR and ALPAKKA_LLAMA_BACKEND when set.
func localConfig(t *testing.T) config.Config {
	t.Helper()
	cfg, err := config.Load(config.DefaultPath())
	if err != nil {
		t.Skipf("config: %v", err)
	}
	if dir := os.Getenv("ALPAKKA_LLAMA_LIB_DIR"); dir != "" {
		cfg.Llama = config.Llama{LibDir: dir, Backend: os.Getenv("ALPAKKA_LLAMA_BACKEND")}
	}
	return cfg
}

func testLlama(t *testing.T) config.Llama {
	t.Helper()
	l := localConfig(t).Llama
	if _, err := os.Stat(l.Binary()); err != nil {
		t.Skip("no llama-server present")
	}
	if _, err := os.Stat(l.BackendDir()); err != nil {
		t.Skip("no llama-server backend directory present")
	}
	return l
}

func smallModel(t *testing.T) *store.Model {
	t.Helper()
	roots := localConfig(t).Store.Roots
	if len(roots) == 0 {
		roots = store.DefaultRoots()
	}
	var sources []store.Source
	for _, r := range roots {
		sources = append(sources, store.NewDir(r))
	}
	models, err := store.NewMulti(nil, sources...).List()
	if err != nil {
		t.Skipf("model roots: %v", err)
	}
	var smallest *store.Model
	for i := range models {
		m := &models[i]
		if m.IsEmbedding() || (smallest != nil && m.Size >= smallest.Size) {
			continue
		}
		smallest = m
	}
	if smallest == nil {
		t.Skip("no chat model in the model roots")
	}
	return smallest
}

func TestArgsCarriesTheSettingsOllamaCannotExpress(t *testing.T) {
	rt := config.Profile{
		SpecType:      strptr("draft-mtp"),
		SpecDraftNMax: intptr(2),
		NumCtx:        intptr(65536),
		CacheTypeK:    strptr("q4_0"),
		CacheTypeV:    strptr("q4_0"),
	}.Runtime("m", "/blob", "/proj", false)

	got := strings.Join(Args(rt, 9999), " ")
	for _, want := range []string{
		"--spec-type draft-mtp",
		"--spec-draft-n-max 2",
		"-c 65536",
		"--cache-type-k q4_0",
		"--jinja",   // reasoning_effort depends on the jinja path
		"--fit off", // no silent downgrade
		"--mmproj /proj",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("args missing %q\ngot: %s", want, got)
		}
	}
}

func TestArgsDisablesProjectorWhenAbsent(t *testing.T) {
	rt := config.Profile{}.Runtime("m", "/blob", "", false)
	got := strings.Join(Args(rt, 1), " ")
	if !strings.Contains(got, "--no-mmproj") {
		t.Errorf("expected --no-mmproj, got: %s", got)
	}
}

func TestWakeRPCNodesSkipsUnconfiguredAddresses(t *testing.T) {
	var logged []string
	s := New(config.Llama{}, map[string]string{
		"192.168.178.62:50052": "aa:bb:cc:dd:ee:ff",
	}, func(format string, args ...any) {
		logged = append(logged, fmt.Sprintf(format, args...))
	})

	s.wakeRPCNodes("192.168.178.62:50052,10.0.0.5:50052")

	if len(logged) != 1 {
		t.Fatalf("logged %d lines, want 1: %v", len(logged), logged)
	}
	if !strings.Contains(logged[0], "192.168.178.62:50052") {
		t.Errorf("log line = %q, want the woken address", logged[0])
	}
}

func TestEnsureLoadsAndServes(t *testing.T) {
	llama := testLlama(t)
	m := smallModel(t)

	s := New(llama, nil, t.Logf)
	s.LoadTimeout = 90 * time.Second
	defer s.Stop()

	rt := config.Profile{NumCtx: intptr(4096)}.Runtime(m.Name, m.ModelPath, "", false)

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	inst, err := s.Ensure(ctx, rt)
	if err != nil {
		t.Fatalf("Ensure: %v", err)
	}

	// The fit check needs llama-server to print the offload line at the supervisor's verbosity.
	fit := inst.Fit()
	if !fit.Seen {
		t.Errorf("no offload line seen at -lv %s; the fit check would be blind.\n%s",
			logVerbosity, inst.Log(30))
	}
	if fit.Seen && !fit.OK() {
		t.Errorf("fit = %d/%d layers on GPU", fit.OffloadedLayers, fit.TotalLayers)
	}

	pid := inst.cmd.Process.Pid

	again, err := s.Ensure(ctx, rt)
	if err != nil {
		t.Fatal(err)
	}
	if again != inst {
		t.Error("identical runtime started a second process")
	}
	again.Release()

	// Give the reference back, since the reload must wait for the instance to be idle.
	inst.Release()

	rt2 := config.Profile{NumCtx: intptr(2048)}.Runtime(m.Name, m.ModelPath, "", false)
	inst2, err := s.Ensure(ctx, rt2)
	if err != nil {
		t.Fatalf("reload: %v", err)
	}
	inst2.Release()
	if inst2 == inst {
		t.Fatal("a changed num_ctx did not trigger a reload")
	}
	if inst2.Runtime().NumCtx != 2048 {
		t.Errorf("reloaded with num_ctx = %d", inst2.Runtime().NumCtx)
	}

	// Trap: a kill that hits a wrapper leaves the real server holding the port and VRAM.
	if alive(pid) {
		t.Errorf("process %d survived the reload", pid)
	}

	s.Stop()
	if alive(inst2.cmd.Process.Pid) {
		t.Errorf("process %d survived Stop", inst2.cmd.Process.Pid)
	}
}

func TestEnsureFailsCleanlyOnMissingModel(t *testing.T) {
	llama := testLlama(t)
	s := New(llama, nil, t.Logf)
	s.LoadTimeout = 20 * time.Second
	defer s.Stop()

	rt := config.Profile{}.Runtime("ghost", "/nonexistent/model.gguf", "", false)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	if _, err := s.Ensure(ctx, rt); err == nil {
		t.Fatal("expected an error for a missing model file")
	}
	if s.Current() != nil {
		t.Error("a failed load left an instance behind")
	}
}

func alive(pid int) bool {
	b, err := os.ReadFile("/proc/" + strconv.Itoa(pid) + "/stat")
	if err != nil {
		return false
	}
	// /proc/<pid>/stat is pid, (comm), state. comm can contain spaces, so the
	// state is read after the closing parenthesis.
	i := strings.LastIndex(string(b), ")")
	if i < 0 {
		return false
	}
	fields := strings.Fields(string(b)[i+1:])
	return len(fields) > 0 && fields[0] != "Z"
}

func strptr(s string) *string { return &s }
func intptr(i int) *int       { return &i }
func boolptr(b bool) *bool    { return &b }

func TestArgsAsksForEmbeddingsOnlyForEmbeddingModels(t *testing.T) {
	chat := strings.Join(Args(config.Profile{}.Runtime("m", "/blob", "", false), 1), " ")
	if strings.Contains(chat, "--embeddings") {
		t.Errorf("a chat model was started for embeddings: %s", chat)
	}
	// Without this llama-server rejects /v1/embeddings, so /api/embed could never work.
	embed := strings.Join(Args(config.Profile{}.Runtime("m", "/blob", "", true), 1), " ")
	if !strings.Contains(embed, "--embeddings") {
		t.Errorf("an embedding model was started without --embeddings: %s", embed)
	}
}

func TestBusyInstanceIsNotEvicted(t *testing.T) {
	i := &Instance{log: newRing(4), notable: newRing(4)}
	i.idle = sync.NewCond(&i.mu)

	i.Touch(time.Millisecond)
	time.Sleep(5 * time.Millisecond)
	if !i.expired(time.Now()) {
		t.Fatal("an idle instance past its keep-alive should be evictable")
	}

	if !i.acquireIfLive() {
		t.Fatal("acquireIfLive on a live instance")
	}
	time.Sleep(5 * time.Millisecond)
	if i.expired(time.Now()) {
		t.Error("an instance with a request in flight was evictable")
	}

	// Releasing restarts the window from completion, the way ollama does.
	i.Touch(time.Minute)
	i.Release()
	if i.expired(time.Now()) {
		t.Error("keep-alive did not restart when the request finished")
	}
	if i.busy() {
		t.Error("still busy after Release")
	}
}

func TestWaitIdleBlocksUntilReleased(t *testing.T) {
	i := &Instance{log: newRing(4), notable: newRing(4)}
	i.idle = sync.NewCond(&i.mu)
	i.acquireIfLive()

	done := make(chan struct{})
	go func() {
		defer close(done)
		if err := i.waitIdle(context.Background()); err != nil {
			t.Errorf("waitIdle: %v", err)
		}
	}()

	select {
	case <-done:
		t.Fatal("waitIdle returned while a request was still in flight")
	case <-time.After(20 * time.Millisecond):
	}

	i.Release()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("waitIdle did not wake when the request finished")
	}
}

func TestWaitIdleHonoursContext(t *testing.T) {
	i := &Instance{log: newRing(4), notable: newRing(4)}
	i.idle = sync.NewCond(&i.mu)
	i.acquireIfLive()

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	if err := i.waitIdle(ctx); err == nil {
		t.Error("waitIdle ignored a cancelled context")
	}
}

func TestCheckFitFailsWhenTheOffloadLineIsMissing(t *testing.T) {
	i := &Instance{log: newRing(4), notable: newRing(4)}
	if err := i.checkFit(); err == nil {
		t.Error("checkFit passed a load it could not verify")
	}
}

func TestCheckFitPermitsOnlyExplicitBoundedPartialOffload(t *testing.T) {
	i := &Instance{
		rt:  config.Runtime{Model: "oversized", NumGPU: 18, AllowPartialOffload: true},
		fit: Fit{Seen: true, OffloadedLayers: 18, TotalLayers: 41},
		log: newRing(4), notable: newRing(4),
	}
	if err := i.checkFit(); err != nil {
		t.Fatalf("explicit partial offload rejected: %v", err)
	}
	i.rt.AllowPartialOffload = false
	if err := i.checkFit(); err == nil {
		t.Fatal("ordinary profile accepted a partial offload")
	}
	i.rt.AllowPartialOffload = true
	i.fit.OffloadedLayers = 19
	if err := i.checkFit(); err == nil {
		t.Fatal("profile exceeded its explicit GPU-layer bound")
	}
}

func TestStderrWriterSplitsAcrossWrites(t *testing.T) {
	i := &Instance{log: newRing(10), notable: newRing(10)}
	w := &stderrWriter{inst: i}
	w.Write([]byte("0.01 E failed to alloc"))
	w.Write([]byte("ate buffer\nload_tensors: offloaded 5/7 layers to GPU\n"))

	if got := i.Log(10); !strings.Contains(got, "failed to allocate buffer") {
		t.Errorf("a line split across writes was mangled: %q", got)
	}
	fit := i.Fit()
	if !fit.Seen || fit.OffloadedLayers != 5 || fit.TotalLayers != 7 {
		t.Errorf("offload line not parsed: %+v", fit)
	}
}

func TestKVBufferIsSummedPerBackendAtItsMaximum(t *testing.T) {
	i := &Instance{log: newRing(10), notable: newRing(10)}
	for _, line := range []string{
		"llama_kv_cache:      ROCm0 KV buffer size =   512.00 MiB",
		"llama_kv_cache:        CPU KV buffer size =    64.00 MiB",
		"llama_kv_cache:      ROCm0 KV buffer size =   512.00 MiB",
	} {
		i.observe(line)
	}
	if got := i.Fit().KVBufferMiB; got != 576 {
		t.Errorf("KVBufferMiB = %v, want 576", got)
	}
}

func TestParseVRAMKiBSumsDRMClients(t *testing.T) {
	got := parseVRAMKiB("drm-memory-vram:\t4096 KiB\ndrm-memory-gtt:\t9000 KiB\ndrm-memory-vram: 2048 KiB\n")
	if got != 6144 {
		t.Fatalf("VRAM KiB = %d, want 6144", got)
	}
}

// llama.cpp's OpenAI embedding endpoint rejects pooling "none", the causal
// default, so the flag has to reach the command line.
func TestArgsCarriesPooling(t *testing.T) {
	rt := config.Profile{Pooling: strptr("last")}.Runtime("m", "/blob", "", true)
	got := strings.Join(Args(rt, 1), " ")
	if !strings.Contains(got, "--pooling last") {
		t.Errorf("args missing --pooling last: %s", got)
	}
	bare := strings.Join(Args(config.Profile{}.Runtime("m", "/blob", "", true), 1), " ")
	if strings.Contains(bare, "--pooling") {
		t.Errorf("pooling forced when unset: %s", bare)
	}
}

func TestArgsCarriesKVStreamArena(t *testing.T) {
	rt := config.Profile{KVStreamArenaMiB: intptr(2048)}.Runtime("m", "/blob", "", false)
	got := strings.Join(Args(rt, 1), " ")
	if !strings.Contains(got, "--kv-stream-arena-mib 2048") {
		t.Errorf("args missing --kv-stream-arena-mib 2048: %s", got)
	}
	bare := strings.Join(Args(config.Profile{}.Runtime("m", "/blob", "", false), 1), " ")
	if strings.Contains(bare, "--kv-stream-arena-mib") {
		t.Errorf("kv stream arena passed when unset: %s", bare)
	}
}

func TestArgsJoinsOverrideTensorIntoOneFlag(t *testing.T) {
	rt := config.Profile{
		NumCPUMoE:      intptr(12),
		OverrideTensor: []string{"a=CPU", "b=CPU"},
	}.Runtime("m", "/blob", "", false)

	args := Args(rt, 1)
	n := 0
	for i, a := range args {
		if a == "--override-tensor" {
			if args[i+1] != "a=CPU,b=CPU" {
				t.Errorf("override-tensor = %q, want %q", args[i+1], "a=CPU,b=CPU")
			}
			n++
		}
	}
	if n != 1 {
		t.Errorf("--override-tensor passed %d times, want 1: %v", n, args)
	}
	if !strings.Contains(strings.Join(args, " "), "--n-cpu-moe 12") {
		t.Errorf("args missing --n-cpu-moe 12: %v", args)
	}

	bare := strings.Join(Args(config.Profile{}.Runtime("m", "/blob", "", false), 1), " ")
	if strings.Contains(bare, "--override-tensor") || strings.Contains(bare, "--n-cpu-moe") {
		t.Errorf("MoE offload forced when unset: %s", bare)
	}
}

func TestArgsCarriesMoEExpertCacheOnlyWhenEnabled(t *testing.T) {
	rt := config.Profile{
		MoEExpertCache:        intptr(8),
		MoEExpertCacheInserts: intptr(2),
	}.Runtime("m", "/blob", "", false)
	got := strings.Join(Args(rt, 1), " ")
	if !strings.Contains(got, "--moe-expert-cache 8") {
		t.Errorf("args missing --moe-expert-cache 8: %s", got)
	}
	if !strings.Contains(got, "--moe-expert-cache-inserts 2") {
		t.Errorf("args missing --moe-expert-cache-inserts 2: %s", got)
	}

	// Inserts alone tunes a cache that is not running, so it stays off the line.
	inserts := config.Profile{MoEExpertCacheInserts: intptr(2)}.Runtime("m", "/blob", "", false)
	if got := strings.Join(Args(inserts, 1), " "); strings.Contains(got, "--moe-expert-cache") {
		t.Errorf("expert cache flags passed with the cache disabled: %s", got)
	}
}

func TestDescribeChangeNamesTheMoESettings(t *testing.T) {
	base := config.Profile{}.Runtime("m", "/blob", "", false)
	for want, rt := range map[string]config.Runtime{
		"num_cpu_moe":      config.Profile{NumCPUMoE: intptr(12)}.Runtime("m", "/blob", "", false),
		"override_tensor":  config.Profile{OverrideTensor: []string{"a=CPU"}}.Runtime("m", "/blob", "", false),
		"moe_expert_cache": config.Profile{MoEExpertCache: intptr(8)}.Runtime("m", "/blob", "", false),
	} {
		if got := describeChange(base, rt); !strings.Contains(got, want) {
			t.Errorf("reload reason does not mention %s: %s", want, got)
		}
	}
}

func TestDescribeChangeNamesTheKVStreamArena(t *testing.T) {
	old := config.Profile{}.Runtime("m", "/blob", "", false)
	new := config.Profile{KVStreamArenaMiB: intptr(2048)}.Runtime("m", "/blob", "", false)
	if got := describeChange(old, new); !strings.Contains(got, "kv_stream_arena") {
		t.Errorf("reload reason does not mention the arena: %s", got)
	}
}

func TestArgsCarriesGPUPlacementOnlyWhenSet(t *testing.T) {
	rt := config.Profile{
		Device:      config.StringList{"Vulkan1", "Vulkan0"},
		TensorSplit: strptr("12,6"),
		SplitMode:   strptr("layer"),
		MainGPU:     intptr(0),
		NoKVOffload: boolptr(true),
	}.Runtime("m", "/blob", "", false)
	got := strings.Join(Args(rt, 1), " ")
	for _, want := range []string{
		"--device Vulkan1,Vulkan0", "--tensor-split 12,6", "--split-mode layer",
		"--main-gpu 0", "--no-kv-offload",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("args missing %q: %s", want, got)
		}
	}
	bare := strings.Join(Args(config.Profile{}.Runtime("m", "/blob", "", false), 1), " ")
	for _, flag := range []string{"--device", "--tensor-split", "--split-mode", "--main-gpu", "--no-kv-offload"} {
		if strings.Contains(bare, flag) {
			t.Errorf("%s passed when unset: %s", flag, bare)
		}
	}
}

func TestDescribeChangeNamesGPUPlacement(t *testing.T) {
	old := config.Profile{}.Runtime("m", "/blob", "", false)
	new := config.Profile{TensorSplit: strptr("3,1")}.Runtime("m", "/blob", "", false)
	if got := describeChange(old, new); !strings.Contains(got, "placement") {
		t.Errorf("reload reason does not mention placement: %s", got)
	}
}

func TestAcquireIfModelReusesOnlyTheSameModel(t *testing.T) {
	i := &Instance{
		rt:      config.Runtime{Model: "qwen", NumCtx: 8192},
		ready:   make(chan struct{}),
		log:     newRing(4),
		notable: newRing(4),
	}
	i.idle = sync.NewCond(&i.mu)
	close(i.ready)
	s := New(config.Llama{}, nil, nil)
	s.cur = i

	if got := s.AcquireIfModel(context.Background(), "qwen", false); got != i {
		t.Fatal("same model with different runtime settings should be reused")
	}
	if !i.busy() {
		t.Error("reuse did not take a reference")
	}
	i.Release()

	if s.AcquireIfModel(context.Background(), "other", false) != nil {
		t.Error("a different model was reused")
	}
	if s.AcquireIfModel(context.Background(), "qwen", true) != nil {
		t.Error("a chat instance was reused for embedding")
	}
}

func TestArgsCarriesBatchDraftCacheAndLoadModeOnlyWhenSet(t *testing.T) {
	rt := config.Profile{
		CacheTypeKDraft: strptr("q4_0"),
		CacheTypeVDraft: strptr("q4_0"),
		NumBatch:        intptr(2048),
		NumUBatch:       intptr(256),
		LoadMode:        strptr("none"),
	}.Runtime("m", "/blob", "", false)
	got := strings.Join(Args(rt, 1), " ")
	for _, want := range []string{
		"--cache-type-k-draft q4_0", "--cache-type-v-draft q4_0",
		"--batch-size 2048", "--ubatch-size 256", "--load-mode none",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("args missing %q: %s", want, got)
		}
	}
	bare := config.Profile{}.Runtime("m", "/blob", "", false)
	for _, flag := range []string{"-draft", "--batch-size", "--ubatch-size", "--load-mode"} {
		if s := strings.Join(Args(bare, 1), " "); strings.Contains(s, flag) {
			t.Errorf("%s passed when unset: %s", flag, s)
		}
	}
	for _, want := range []string{"draft cache", "batch", "load_mode"} {
		if d := describeChange(bare, rt); !strings.Contains(d, want) {
			t.Errorf("reload reason does not mention %s: %s", want, d)
		}
	}
}
