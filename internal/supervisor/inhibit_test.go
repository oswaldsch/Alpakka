package supervisor

import (
	"os/exec"
	"testing"
)

func TestInhibitorStartsAndStops(t *testing.T) {
	if _, err := exec.LookPath("systemd-inhibit"); err != nil {
		t.Skip("systemd-inhibit not on PATH")
	}

	cmd, err := startInhibitor("test-model")
	if err != nil {
		t.Fatalf("startInhibitor: %v", err)
	}
	if cmd.Process == nil {
		t.Fatal("expected a running process")
	}

	stopInhibitor(cmd)

	// ProcessState.Exited() is false for a SIGTERM exit, so this only checks
	// that Wait returned rather than that the exit was "clean".
	if cmd.ProcessState == nil {
		t.Error("expected the inhibitor process to have been reaped")
	}
}

func TestStopInhibitorIsNilSafe(t *testing.T) {
	stopInhibitor(nil)
	stopInhibitor(&exec.Cmd{})
}
