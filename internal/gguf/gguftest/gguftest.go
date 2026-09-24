// Package gguftest writes GGUF headers for tests.
package gguftest

import (
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"sort"
	"testing"
)

// Only the header is written, since the reader never touches the weights.
func Write(t testing.TB, path string, kv map[string]any) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, Bytes(kv), 0o644); err != nil {
		t.Fatal(err)
	}
}

func Bytes(kv map[string]any) []byte {
	var b bytes.Buffer
	b.Write([]byte("GGUF"))
	put32(&b, 3)
	put64(&b, 0) // tensor count
	put64(&b, uint64(len(kv)))

	keys := make([]string, 0, len(kv))
	for k := range kv {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		putStr(&b, k)
		switch v := kv[k].(type) {
		case string:
			put32(&b, 8)
			putStr(&b, v)
		case uint32:
			put32(&b, 4)
			put32(&b, v)
		case bool:
			put32(&b, 7)
			var n byte
			if v {
				n = 1
			}
			b.WriteByte(n)
		default:
			panic("gguftest: unsupported value type")
		}
	}
	return b.Bytes()
}

func put32(b *bytes.Buffer, v uint32) {
	var raw [4]byte
	binary.LittleEndian.PutUint32(raw[:], v)
	b.Write(raw[:])
}

func put64(b *bytes.Buffer, v uint64) {
	var raw [8]byte
	binary.LittleEndian.PutUint64(raw[:], v)
	b.Write(raw[:])
}

func putStr(b *bytes.Buffer, s string) {
	put64(b, uint64(len(s)))
	b.WriteString(s)
}
