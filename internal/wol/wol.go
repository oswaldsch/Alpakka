// Package wol sends Wake-on-LAN magic packets, so a load that offloads onto an
// RPC node does not need someone to walk over and power the box back on.
package wol

import (
	"fmt"
	"net"
	"syscall"
)

// Wake sends a magic packet for mac to the LAN broadcast address. rpc-server
// nodes are LAN-only by the same rule --rpc itself is held to, so a broadcast
// is always in reach.
func Wake(mac string) error {
	hw, err := net.ParseMAC(mac)
	if err != nil {
		return fmt.Errorf("wol: %w", err)
	}
	if len(hw) != 6 {
		return fmt.Errorf("wol: %q: not a 6-byte MAC address", mac)
	}

	conn, err := net.ListenPacket("udp4", ":0")
	if err != nil {
		return fmt.Errorf("wol: %w", err)
	}
	defer conn.Close()

	// Linux refuses sendto on a broadcast address unless SO_BROADCAST is set on
	// the socket; net.ListenPacket does not set it.
	sc, ok := conn.(syscall.Conn)
	if !ok {
		return fmt.Errorf("wol: %T does not expose its socket", conn)
	}
	raw, err := sc.SyscallConn()
	if err != nil {
		return fmt.Errorf("wol: %w", err)
	}
	var sockErr error
	if err := raw.Control(func(fd uintptr) {
		sockErr = syscall.SetsockoptInt(int(fd), syscall.SOL_SOCKET, syscall.SO_BROADCAST, 1)
	}); err != nil {
		return fmt.Errorf("wol: %w", err)
	}
	if sockErr != nil {
		return fmt.Errorf("wol: %w", sockErr)
	}

	dst, err := net.ResolveUDPAddr("udp4", "255.255.255.255:9")
	if err != nil {
		return fmt.Errorf("wol: %w", err)
	}
	if _, err := conn.WriteTo(magicPacket(hw), dst); err != nil {
		return fmt.Errorf("wol: %w", err)
	}
	return nil
}

// magicPacket is six 0xFF bytes followed by the target MAC repeated sixteen
// times, the payload every Wake-on-LAN listener expects.
func magicPacket(hw net.HardwareAddr) []byte {
	packet := make([]byte, 0, 102)
	for range 6 {
		packet = append(packet, 0xFF)
	}
	for range 16 {
		packet = append(packet, hw...)
	}
	return packet
}
