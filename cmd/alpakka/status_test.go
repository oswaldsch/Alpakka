package main

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"

	"github.com/oswaldsch/alpakka/internal/config"
)

func TestPrintListShowsNameQuantSizeAndAge(t *testing.T) {
	now := time.Date(2026, 9, 23, 12, 0, 0, 0, time.UTC)
	var out bytes.Buffer
	printList(&out, api.ListResponse{Models: []api.ListModelResponse{{
		Name:       "qwen3.5-mtp:35b",
		Size:       23_600_000_000,
		ModifiedAt: now.Add(-3 * 24 * time.Hour),
		Details:    api.ModelDetails{QuantizationLevel: "Q4_K_M"},
	}}}, now)
	for _, want := range []string{"NAME", "qwen3.5-mtp:35b", "Q4_K_M", "23.6 GB", "3 days ago"} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("list output missing %q:\n%s", want, out.String())
		}
	}
}

func TestPrintPSShowsVRAMContextAndExpiry(t *testing.T) {
	now := time.Date(2026, 9, 23, 12, 0, 0, 0, time.UTC)
	var out bytes.Buffer
	printPS(&out, psResponse{Models: []psEntry{
		{ProcessModelResponse: api.ProcessModelResponse{Name: "a:1", SizeVRAM: 16_234_610_130, ContextLength: 131072, ExpiresAt: now.Add(4 * time.Minute)}},
		{ProcessModelResponse: api.ProcessModelResponse{Name: "b:1", ExpiresAt: time.Time{}}},
	}}, false, now)
	for _, want := range []string{"a:1", "16.2 GB", "131072", "in 4m0s", "forever"} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("ps output missing %q:\n%s", want, out.String())
		}
	}
}

func TestPrintPSFoldsInTheProcessStatus(t *testing.T) {
	now := time.Date(2026, 9, 23, 12, 0, 0, 0, time.UTC)
	var e psEntry
	e.Name = "qwen:q3"
	e.Alpakka.Busy = true
	e.Alpakka.StartedAt = now.Add(-3 * time.Minute)
	e.Alpakka.Fit.OffloadedLayers, e.Alpakka.Fit.TotalLayers = 65, 65
	e.Alpakka.Fit.KVBufferMiB = 1088
	e.Alpakka.Fit.CPUBufferMiB = 512
	e.Alpakka.Runtime = config.Runtime{Model: "qwen:q3", ModelPath: "/m.gguf", NumCtx: 32768, SpecType: "draft-mtp", MainGPU: -1}

	var out bytes.Buffer
	printPS(&out, psResponse{Models: []psEntry{e}}, false, now)
	for _, want := range []string{"GPU LAYERS", "65/65", "generating"} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("ps output missing %q:\n%s", want, out.String())
		}
	}
	if strings.Contains(out.String(), "KV cache") {
		t.Errorf("detail printed without -v:\n%s", out.String())
	}

	out.Reset()
	printPS(&out, psResponse{Models: []psEntry{e}}, true, now)
	for _, want := range []string{"up 3m0s", "1088 MiB", "weights on CPU", "num_ctx=32768 spec_type=draft-mtp"} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("ps -v output missing %q:\n%s", want, out.String())
		}
	}
	for _, unwanted := range []string{"model_path", "main_gpu", "flash_attn="} {
		if strings.Contains(out.String(), unwanted) {
			t.Errorf("ps -v printed unset or redundant %q:\n%s", unwanted, out.String())
		}
	}
}

func TestListAndPSReadTheRunningServer(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/tags":
			w.Write([]byte(`{"models":[{"name":"m:1"}]}`))
		case "/api/ps":
			w.Write([]byte(`{"models":[]}`))
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	host := strings.TrimPrefix(srv.URL, "http://")

	var tags api.ListResponse
	if err := getJSON(host, "/api/tags", &tags); err != nil || len(tags.Models) != 1 {
		t.Fatalf("tags = %+v, err %v", tags, err)
	}
	if err := run([]string{"ps", "-host", host}); err != nil {
		t.Fatalf("ps: %v", err)
	}
	if err := getJSON(host, "/nope", &tags); err == nil {
		t.Error("a non-200 answer was accepted")
	}
}
