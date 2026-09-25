package api

import (
	"encoding/json"
	"net/http"
	"strings"
	"testing"
)

func TestUnloadWithNothingResident(t *testing.T) {
	h := testServer(t)

	w := do(t, h, http.MethodPost, "/alpakka/unload", "")
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (body: %s)", w.Code, w.Body.String())
	}
	var got unloadResponse
	if err := json.Unmarshal(w.Body.Bytes(), &got); err != nil || got.Model != "" {
		t.Errorf("body = %s, want no model", w.Body.String())
	}

	w = do(t, h, http.MethodPost, "/alpakka/unload", `{"model":"qwen3:0.6b"}`)
	if w.Code != http.StatusNotFound || !strings.Contains(w.Body.String(), "not loaded") {
		t.Errorf("named model: status = %d, body %s; want 404 not loaded", w.Code, w.Body.String())
	}
}

func TestLogsWithNothingStarted(t *testing.T) {
	w := do(t, testServer(t), http.MethodGet, "/alpakka/logs?n=5", "")
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d (body: %s)", w.Code, w.Body.String())
	}
	var got logsResponse
	if err := json.Unmarshal(w.Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Model != "" || got.Running || got.Lines == nil || len(got.Lines) != 0 {
		t.Errorf("body = %s, want an empty log", w.Body.String())
	}
}

func TestLogsRejectsABadLineCount(t *testing.T) {
	for _, n := range []string{"0", "-3", "lots"} {
		if w := do(t, testServer(t), http.MethodGet, "/alpakka/logs?n="+n, ""); w.Code != http.StatusBadRequest {
			t.Errorf("n=%s: status = %d, want 400", n, w.Code)
		}
	}
}
