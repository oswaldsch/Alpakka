package api

import (
	"errors"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/oswaldsch/alpakka/internal/supervisor"
)

type unloadRequest struct {
	Model string `json:"model"`
	// Stops the process even while it streams, which cuts that response off.
	Force bool `json:"force"`
}

type unloadResponse struct {
	// Empty when nothing was resident.
	Model string `json:"model"`
}

// Ollama unloads through a generate request with keep_alive 0, which here would load the
// model first to answer it. This stops the process without that detour.
func (s *Server) handleUnload(w http.ResponseWriter, r *http.Request) {
	var req unloadRequest
	if r.ContentLength != 0 && !decode(w, r, &req) {
		return
	}
	name := req.Model
	if name != "" {
		// A bare name or one of several spellings has to match the resident runtime's canonical name.
		if m, err := s.Store.Get(name); err == nil {
			name = m.Name
		}
	}

	// Serialized with loads, so a request cannot start one between the check and the stop.
	s.loadMu.Lock()
	defer s.loadMu.Unlock()

	unloaded, err := s.Super.Unload(r.Context(), name, req.Force)
	switch {
	case errors.Is(err, supervisor.ErrNotLoaded):
		writeError(w, http.StatusNotFound, "model '"+req.Model+"' is not loaded")
	case err != nil:
		writeError(w, http.StatusInternalServerError, err.Error())
	default:
		writeJSON(w, http.StatusOK, unloadResponse{Model: unloaded})
	}
}

const (
	logsDefaultLines = 100
	// The supervisor keeps no more than this per process.
	logsMaxLines = 400
)

type logsResponse struct {
	Model     string    `json:"model,omitempty"`
	Running   bool      `json:"running"`
	StartedAt time.Time `json:"started_at,omitzero"`
	Lines     []string  `json:"lines"`
}

// The last process's stderr, kept after it exits so a failed load can be diagnosed after the fact.
func (s *Server) handleLogs(w http.ResponseWriter, r *http.Request) {
	n := logsDefaultLines
	if v := r.URL.Query().Get("n"); v != "" {
		parsed, err := strconv.Atoi(v)
		if err != nil || parsed < 1 {
			writeError(w, http.StatusBadRequest, "n must be a positive number of lines")
			return
		}
		n = min(parsed, logsMaxLines)
	}

	resp := logsResponse{Lines: []string{}}
	if inst := s.Super.Last(); inst != nil {
		resp.Model = inst.Runtime().Model
		resp.Running = inst.Running()
		resp.StartedAt = inst.StartedAt()
		if tail := inst.Log(n); tail != "" {
			resp.Lines = strings.Split(tail, "\n")
		}
	}
	writeJSON(w, http.StatusOK, resp)
}
