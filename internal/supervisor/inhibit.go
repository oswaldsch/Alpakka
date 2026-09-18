package supervisor

import (
	"os/exec"
	"syscall"
)

// startInhibitor blocks system sleep and idle suspend for as long as the
// returned process runs.
func startInhibitor(model string) (*exec.Cmd, error) {
	cmd := exec.Command("systemd-inhibit",
		"--what=sleep:idle",
		"--who=alpakka",
		"--why=model "+model+" is offloaded to an RPC node",
		"--mode=block",
		"sleep", "infinity")
	// Its own process group: SIGTERM has to reach the sleep, not just
	// systemd-inhibit, because the inhibit lock is released when that sleep
	// exits and systemd-inhibit does not forward signals to it.
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	return cmd, nil
}

// stopInhibitor releases an inhibitor started by startInhibitor.
func stopInhibitor(cmd *exec.Cmd) {
	if cmd == nil || cmd.Process == nil {
		return
	}
	pgid, err := syscall.Getpgid(cmd.Process.Pid)
	if err != nil {
		pgid = cmd.Process.Pid
	}
	_ = syscall.Kill(-pgid, syscall.SIGTERM)
	_ = cmd.Wait()
}
