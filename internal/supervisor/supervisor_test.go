package supervisor

import (
	"context"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/oswald/alpakka/internal/config"
	"github.com/oswald/alpakka/internal/store"
)

func testLlama(t *testing.T) config.Llama {
	t.Helper()
	l := config.Llama{LibDir: "/usr/local/lib/ollama", Backend: "rocm_v7_2"}
	if _, err := os.Stat(l.Binary()); err != nil {
		t.Skip("llama-server not present")
	}
	if _, err := os.Stat(l.BackendDir()); err != nil {
		t.Skip("rocm backend not present")
	}
	return l
}

// smallModel resolves a model small enough to load repeatedly in a test.
func smallModel(t *testing.T) *store.Model {
	t.Helper()
	root := store.DefaultRoot()
	if _, err := os.Stat(root); err != nil {
		t.Skip("ollama model store not present")
	}
	m, err := store.New(root).Get("qwen3:0.6b")
	if err != nil {
		t.Skipf("qwen3:0.6b not pulled: %v", err)
	}
	return m
}

func TestArgsCarriesTheSettingsOllamaCannotExpress(t *testing.T) {
	rt := config.Profile{
		SpecType:      strptr("draft-mtp"),
		SpecDraftNMax: intptr(2),
		NumCtx:        intptr(65536),
		CacheTypeK:    strptr("q4_0"),
		CacheTypeV:    strptr("q4_0"),
	}.Runtime("m", "/blob", "/proj")

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
	rt := config.Profile{}.Runtime("m", "/blob", "")
	got := strings.Join(Args(rt, 1), " ")
	if !strings.Contains(got, "--no-mmproj") {
		t.Errorf("expected --no-mmproj, got: %s", got)
	}
}

// TestEnsureLoadsAndServes is the end-to-end supervisor check: a real
// llama-server comes up, the fit check reads a real offload line, and the
// process is really gone afterwards.
func TestEnsureLoadsAndServes(t *testing.T) {
	llama := testLlama(t)
	m := smallModel(t)

	s := New(llama, t.Logf)
	s.LoadTimeout = 90 * time.Second
	defer s.Stop()

	rt := config.Profile{NumCtx: intptr(4096)}.Runtime(m.Name, m.ModelPath, "")

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	inst, err := s.Ensure(ctx, rt)
	if err != nil {
		t.Fatalf("Ensure: %v", err)
	}

	// The fit check is only meaningful if llama-server actually printed the
	// offload line at the verbosity the supervisor asks for.
	fit := inst.Fit()
	if !fit.Seen {
		t.Errorf("no offload line seen at -lv %s; the fit check would be blind.\n%s",
			logVerbosity, inst.Log(30))
	}
	if fit.Seen && !fit.OK() {
		t.Errorf("fit = %d/%d layers on GPU", fit.OffloadedLayers, fit.TotalLayers)
	}

	pid := inst.cmd.Process.Pid

	// The same runtime must reuse the process rather than reload.
	again, err := s.Ensure(ctx, rt)
	if err != nil {
		t.Fatal(err)
	}
	if again != inst {
		t.Error("identical runtime started a second process")
	}

	// A changed process-level flag must reload rather than silently continue.
	rt2 := config.Profile{NumCtx: intptr(2048)}.Runtime(m.Name, m.ModelPath, "")
	inst2, err := s.Ensure(ctx, rt2)
	if err != nil {
		t.Fatalf("reload: %v", err)
	}
	if inst2 == inst {
		t.Fatal("a changed num_ctx did not trigger a reload")
	}
	if inst2.Runtime().NumCtx != 2048 {
		t.Errorf("reloaded with num_ctx = %d", inst2.Runtime().NumCtx)
	}

	// Trap: a kill that hits a wrapper leaves the real server holding the port
	// and the VRAM. Verify the original process is genuinely gone.
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
	s := New(llama, t.Logf)
	s.LoadTimeout = 20 * time.Second
	defer s.Stop()

	rt := config.Profile{}.Runtime("ghost", "/nonexistent/model.gguf", "")
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	if _, err := s.Ensure(ctx, rt); err == nil {
		t.Fatal("expected an error for a missing model file")
	}
	if s.Current() != nil {
		t.Error("a failed load left an instance behind")
	}
}

// alive reports whether a pid is a live process rather than a reaped zombie.
func alive(pid int) bool {
	b, err := os.ReadFile("/proc/" + strconv.Itoa(pid) + "/stat")
	if err != nil {
		return false
	}
	// /proc/<pid>/stat: pid, (comm), state, ...  comm can contain spaces, so
	// the state field is read from after the closing parenthesis.
	i := strings.LastIndex(string(b), ")")
	if i < 0 {
		return false
	}
	fields := strings.Fields(string(b)[i+1:])
	return len(fields) > 0 && fields[0] != "Z"
}

func strptr(s string) *string { return &s }
func intptr(i int) *int       { return &i }
