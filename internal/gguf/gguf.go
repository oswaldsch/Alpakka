// Package gguf reads the metadata header of a GGUF file.
//
// Only the key/value block is parsed; tensor data is never touched. Ollama
// surfaces this metadata as details.family, parameter_size, context_length and
// the model_info map, so alpakka has to read it to answer /api/tags and
// /api/show faithfully.
package gguf

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"strings"
)

const magic = 0x46554747 // "GGUF" little-endian

// value types, per the GGUF spec
const (
	typeUint8 uint32 = iota
	typeInt8
	typeUint16
	typeInt16
	typeUint32
	typeInt32
	typeFloat32
	typeBool
	typeString
	typeArray
	typeUint64
	typeInt64
	typeFloat64
)

// maxArrayLen caps how many array elements are materialised. Token vocabularies
// run to hundreds of thousands of entries and nothing here needs them; longer
// arrays are walked and discarded so parsing can continue to the next key.
const maxArrayLen = 1024

// File is the parsed metadata header of a GGUF file.
type File struct {
	Version     uint32
	TensorCount uint64
	KV          map[string]any
	Tensors     []Tensor
}

// Tensor is one entry of the tensor-info block: name and shape only. Offsets
// and data are not read.
type Tensor struct {
	Name  string
	Shape []uint64
	Type  uint32
}

// Elements is the number of values in the tensor.
func (t Tensor) Elements() uint64 {
	n := uint64(1)
	for _, d := range t.Shape {
		n *= d
	}
	return n
}

// ParameterCount sums the elements of every tensor. GGUFs converted by some
// toolchains omit general.parameter_count, and this is what ollama reports as
// details.parameter_size, so it is always computed rather than read.
func (f *File) ParameterCount() uint64 {
	var n uint64
	for _, t := range f.Tensors {
		n += t.Elements()
	}
	return n
}

// Architecture returns general.architecture, e.g. "qwen35".
func (f *File) Architecture() string {
	s, _ := f.KV["general.architecture"].(string)
	return s
}

// ArchKV looks up a key under the file's architecture namespace, so callers can
// ask for "context_length" without knowing it is stored as "qwen35.context_length".
func (f *File) ArchKV(suffix string) (any, bool) {
	v, ok := f.KV[f.Architecture()+"."+suffix]
	return v, ok
}

// Uint reads an integer-valued key, normalising across the GGUF integer types.
func (f *File) Uint(key string) (uint64, bool) {
	return toUint(f.KV[key])
}

// ArchUint reads an integer-valued key under the architecture namespace.
func (f *File) ArchUint(suffix string) (uint64, bool) {
	v, ok := f.ArchKV(suffix)
	if !ok {
		return 0, false
	}
	return toUint(v)
}

// String reads a string-valued key.
func (f *File) String(key string) (string, bool) {
	s, ok := f.KV[key].(string)
	return s, ok
}

func toUint(v any) (uint64, bool) {
	switch n := v.(type) {
	case uint8:
		return uint64(n), true
	case uint16:
		return uint64(n), true
	case uint32:
		return uint64(n), true
	case uint64:
		return n, true
	case int8:
		return uint64(n), true
	case int16:
		return uint64(n), true
	case int32:
		return uint64(n), true
	case int64:
		return uint64(n), true
	}
	return 0, false
}

// Open parses the metadata header of the GGUF file at path.
func Open(path string) (*File, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return Read(bufio.NewReaderSize(f, 1<<20))
}

// Read parses a GGUF metadata header from r, stopping after the last key.
func Read(r io.Reader) (*File, error) {
	d := &decoder{r: r}

	if got := d.u32(); got != magic {
		return nil, fmt.Errorf("gguf: bad magic %#x", got)
	}
	out := &File{Version: d.u32(), KV: map[string]any{}}
	if out.Version < 2 || out.Version > 3 {
		return nil, fmt.Errorf("gguf: unsupported version %d", out.Version)
	}
	out.TensorCount = d.u64()
	n := d.u64()
	if d.err != nil {
		return nil, d.err
	}
	if n > 1<<20 {
		return nil, fmt.Errorf("gguf: implausible metadata count %d", n)
	}

	for i := uint64(0); i < n; i++ {
		key := d.str()
		v := d.value(d.u32())
		if d.err != nil {
			return nil, fmt.Errorf("gguf: reading key %d (%q): %w", i, key, d.err)
		}
		if v != nil {
			out.KV[key] = v
		}
	}

	if out.TensorCount > 1<<20 {
		return nil, fmt.Errorf("gguf: implausible tensor count %d", out.TensorCount)
	}
	out.Tensors = make([]Tensor, 0, out.TensorCount)
	for i := uint64(0); i < out.TensorCount; i++ {
		t := Tensor{Name: d.str()}
		nd := d.u32()
		if d.err != nil {
			return nil, fmt.Errorf("gguf: reading tensor %d: %w", i, d.err)
		}
		if nd > 8 {
			return nil, fmt.Errorf("gguf: tensor %q has %d dimensions", t.Name, nd)
		}
		t.Shape = make([]uint64, nd)
		for j := range t.Shape {
			t.Shape[j] = d.u64()
		}
		t.Type = d.u32()
		d.u64() // offset, unused
		if d.err != nil {
			return nil, fmt.Errorf("gguf: reading tensor %d (%q): %w", i, t.Name, d.err)
		}
		out.Tensors = append(out.Tensors, t)
	}
	return out, nil
}

type decoder struct {
	r   io.Reader
	err error
	buf [8]byte
}

func (d *decoder) read(n int) []byte {
	if d.err != nil {
		return nil
	}
	b := d.buf[:n]
	if _, err := io.ReadFull(d.r, b); err != nil {
		d.err = err
		return nil
	}
	return b
}

func (d *decoder) u8() uint8 {
	b := d.read(1)
	if b == nil {
		return 0
	}
	return b[0]
}

func (d *decoder) u16() uint16 {
	b := d.read(2)
	if b == nil {
		return 0
	}
	return binary.LittleEndian.Uint16(b)
}

func (d *decoder) u32() uint32 {
	b := d.read(4)
	if b == nil {
		return 0
	}
	return binary.LittleEndian.Uint32(b)
}

func (d *decoder) u64() uint64 {
	b := d.read(8)
	if b == nil {
		return 0
	}
	return binary.LittleEndian.Uint64(b)
}

func (d *decoder) str() string {
	n := d.u64()
	if d.err != nil {
		return ""
	}
	if n > 1<<28 {
		d.err = fmt.Errorf("gguf: implausible string length %d", n)
		return ""
	}
	b := make([]byte, n)
	if _, err := io.ReadFull(d.r, b); err != nil {
		d.err = err
		return ""
	}
	return string(b)
}

// value decodes one value of the given type. It returns nil for values that are
// skipped rather than kept (over-long arrays), which the caller drops.
func (d *decoder) value(t uint32) any {
	switch t {
	case typeUint8:
		return d.u8()
	case typeInt8:
		return int8(d.u8())
	case typeUint16:
		return d.u16()
	case typeInt16:
		return int16(d.u16())
	case typeUint32:
		return d.u32()
	case typeInt32:
		return int32(d.u32())
	case typeFloat32:
		return math.Float32frombits(d.u32())
	case typeBool:
		return d.u8() != 0
	case typeString:
		return d.str()
	case typeUint64:
		return d.u64()
	case typeInt64:
		return int64(d.u64())
	case typeFloat64:
		return math.Float64frombits(d.u64())
	case typeArray:
		return d.array()
	}
	d.err = fmt.Errorf("gguf: unknown value type %d", t)
	return nil
}

func (d *decoder) array() any {
	et := d.u32()
	n := d.u64()
	if d.err != nil {
		return nil
	}
	keep := n <= maxArrayLen
	var out []any
	if keep {
		out = make([]any, 0, n)
	}
	for i := uint64(0); i < n; i++ {
		v := d.value(et)
		if d.err != nil {
			return nil
		}
		if keep {
			out = append(out, v)
		}
	}
	if !keep {
		return nil
	}
	return out
}

// FileTypeName maps general.file_type to the quantization label ollama reports,
// e.g. 12 -> "Q3_K_M". Unknown values report as "unknown", matching ollama.
func FileTypeName(ft uint64) string {
	names := map[uint64]string{
		0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1",
		10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S",
		15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS",
		20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S",
		25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S", 29: "IQ2_M",
		30: "IQ4_XS", 31: "IQ1_M", 32: "BF16", 36: "TQ1_0", 37: "TQ2_0",
	}
	if s, ok := names[ft]; ok {
		return s
	}
	return "unknown"
}

// HumanParams formats a parameter count the way ollama does, e.g. "27.3B".
func HumanParams(n uint64) string {
	switch {
	case n == 0:
		return ""
	case n >= 1e12:
		return trimZero(float64(n)/1e12) + "T"
	case n >= 1e9:
		return trimZero(float64(n)/1e9) + "B"
	case n >= 1e6:
		return trimZero(float64(n)/1e6) + "M"
	default:
		return trimZero(float64(n)/1e3) + "K"
	}
}

func trimZero(f float64) string {
	s := fmt.Sprintf("%.1f", f)
	return strings.TrimSuffix(s, ".0")
}
