package wol

import (
	"bytes"
	"net"
	"testing"
)

func TestMagicPacketShape(t *testing.T) {
	hw, err := net.ParseMAC("aa:bb:cc:dd:ee:ff")
	if err != nil {
		t.Fatal(err)
	}
	got := magicPacket(hw)
	if len(got) != 102 {
		t.Fatalf("len = %d, want 102", len(got))
	}
	if !bytes.Equal(got[:6], []byte{0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF}) {
		t.Errorf("header = % x, want six 0xFF bytes", got[:6])
	}
	for i := range 16 {
		if !bytes.Equal(got[6+i*6:6+i*6+6], hw) {
			t.Errorf("repetition %d = % x, want %x", i, got[6+i*6:6+i*6+6], hw)
		}
	}
}

func TestWakeRejectsInvalidMAC(t *testing.T) {
	if err := Wake("not-a-mac"); err == nil {
		t.Fatal("expected an error for an invalid MAC address")
	}
}

func TestWakeRejectsEUI64(t *testing.T) {
	if err := Wake("aa:bb:cc:ff:fe:dd:ee:ff"); err == nil {
		t.Fatal("expected an error for an 8-byte EUI-64 address")
	}
}
