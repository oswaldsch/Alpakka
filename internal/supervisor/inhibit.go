package supervisor

import (
	"os/exec"
	"syscall"
)

func startInhibitor(model string) (*exec.Cmd, error) {
	cmd := exec.Command("systemd-inhibit",
		"--what=sleep:idle",
		"--who=alpakka",
		"--why=model "+model+" is offloaded to an RPC node",
		"--mode=block",
		"sleep", "infinity")
	// Own process group so SIGTERM reaches the sleep. systemd-inhibit does not
	// forward signals to it and the lock is released when the sleep exits.
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	return cmd, nil
}

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
