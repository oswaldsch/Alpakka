package api

import (
	"fmt"
	"math/rand"
	"net/http"
	"strings"
	"time"

	"github.com/ollama/ollama/api"

	"github.com/oswald/alpakka/internal/config"
	"github.com/oswald/alpakka/internal/supervisor"
	"github.com/oswald/alpakka/internal/translate"
)

// benchRequest is one measurement of one configuration.
//
// Options is ollama's options object, so every process-level setting alpakka
// understands is benchmarkable through the vocabulary it is already configured
// in, and a setting added there needs nothing here.
type benchRequest struct {
	Model        string         `json:"model"`
	Options      map[string]any `json:"options"`
	Prompt       string         `json:"prompt"`
	PromptTokens int            `json:"prompt_tokens"`
	Runs         int            `json:"runs"`
	IgnoreEOS    *bool          `json:"ignore_eos"`
	KeepAlive    *api.Duration  `json:"keep_alive"`
}

type benchResponse struct {
	Model   string         `json:"model"`
	Options map[string]any `json:"options,omitempty"`
	// Runtime is what llama-server is actually running, which is not
	// necessarily what was asked for: a request only reloads settings the
	// profile does not already carry.
	Runtime config.Runtime `json:"runtime"`
	Fit     benchFit       `json:"fit"`
	// LoadMS is what this request spent getting the process ready. Reloaded
	// says whether it paid for the load itself, which is the cost a
	// configuration that reloads on every request keeps paying.
	LoadMS    float64    `json:"load_ms"`
	Reloaded  bool       `json:"reloaded"`
	Runs      []benchRun `json:"runs"`
	Aggregate *benchRun  `json:"aggregate,omitempty"`
	Sample    string     `json:"sample"`
}

// benchRun holds llama.cpp's own timings for one generation. The rates are
// derived from those and not from wall clock, which would fold in the round
// trip and the SSE parse.
type benchRun struct {
	PromptTokens  int     `json:"prompt_tokens"`
	PromptMS      float64 `json:"prompt_ms"`
	PromptTPS     float64 `json:"prompt_tps"`
	PredictTokens int     `json:"predict_tokens"`
	PredictMS     float64 `json:"predict_ms"`
	PredictTPS    float64 `json:"predict_tps"`
	WallMS        float64 `json:"wall_ms"`
	DoneReason    string  `json:"done_reason,omitempty"`
}

type benchFit struct {
	OffloadedLayers int     `json:"offloaded_layers"`
	TotalLayers     int     `json:"total_layers"`
	CPUBufferMiB    float64 `json:"cpu_buffer_mib"`
	KVBufferMiB     float64 `json:"kv_buffer_mib"`
	VRAMMiB         float64 `json:"vram_mib,omitempty"`
}

const (
	benchDefaultPromptTokens = 512
	benchDefaultPredict      = 128
	benchMaxRuns             = 20
	benchMaxPromptTokens     = 1 << 20
	benchSampleChars         = 240
)

func (s *Server) handleBench(w http.ResponseWriter, r *http.Request) {
	var req benchRequest
	if !decode(w, r, &req) {
		return
	}
	if req.Model == "" {
		writeError(w, http.StatusBadRequest, "model is required")
		return
	}
	if req.Prompt != "" && req.PromptTokens > 0 {
		writeError(w, http.StatusBadRequest, "give either prompt or prompt_tokens, not both")
		return
	}
	if req.PromptTokens < 0 || req.PromptTokens > benchMaxPromptTokens {
		writeError(w, http.StatusBadRequest, fmt.Sprintf(
			"prompt_tokens must be between 0 and %d", benchMaxPromptTokens))
		return
	}
	if req.Runs < 0 || req.Runs > benchMaxRuns {
		writeError(w, http.StatusBadRequest, fmt.Sprintf("runs must be between 1 and %d", benchMaxRuns))
		return
	}
	runs := req.Runs
	if runs == 0 {
		runs = 1
	}
	if req.Prompt == "" && req.PromptTokens == 0 {
		req.PromptTokens = benchDefaultPromptTokens
	}

	arrived := time.Now()
	inst, profile, load, err := s.resolve(r.Context(), req.Model, req.Options, req.KeepAlive, false)
	if err != nil {
		writeResolveError(w, req.Model, err)
		return
	}
	defer inst.Release()

	// A benchmark needs a token budget it chose, so an unset or unlimited
	// num_predict becomes a concrete one rather than running to the model's
	// own stopping point.
	if profile.NumPredict == nil || *profile.NumPredict <= 0 {
		profile.NumPredict = ptr(benchDefaultPredict)
	}

	out := benchResponse{
		Model:    inst.Runtime().Model,
		Options:  req.Options,
		Runtime:  inst.Runtime(),
		Fit:      fitOf(inst.Fit()),
		LoadMS:   msOf(load),
		Reloaded: inst.StartedAt().After(arrived),
		Runs:     make([]benchRun, 0, runs),
	}

	for i := 0; i < runs; i++ {
		prompt := req.Prompt
		if prompt == "" {
			// A fresh prompt per run, and words drawn at random rather than
			// repeated: predictable filler would let a draft model verify
			// several tokens a pass and report a decode rate no real workload
			// reaches.
			prompt = fillerPrompt(req.PromptTokens, int64(i))
		}
		run, sample, err := s.benchOnce(r, inst, profile, prompt, req.IgnoreEOS == nil || *req.IgnoreEOS)
		if err != nil {
			var up errUpstream
			if errorsAs(err, &up) {
				writeError(w, up.status, up.Error())
			} else {
				writeError(w, http.StatusBadGateway, err.Error())
			}
			return
		}
		out.Runs = append(out.Runs, run)
		out.Sample = sample
	}
	summary := out.Runs[len(out.Runs)-1]
	if runs > 1 {
		out.Aggregate = ptr(aggregate(out.Runs))
		summary = *out.Aggregate
	}

	s.logf("bench %s: %d run(s), %.1f tok/s prefill, %.1f tok/s decode",
		out.Model, runs, summary.PromptTPS, summary.PredictTPS)
	writeJSON(w, http.StatusOK, out)
}

func (s *Server) benchOnce(r *http.Request, inst *supervisor.Instance, profile config.Profile,
	prompt string, ignoreEOS bool) (benchRun, string, error) {

	body := translate.BuildRequest(inst.Runtime().Model,
		[]translate.Message{{Role: "user", Content: prompt}}, profile, true)
	body.CachePrompt = ptr(false)
	if ignoreEOS {
		body.IgnoreEOS = ptr(true)
	}

	start := time.Now()
	resp, err := s.post(r, inst.BaseURL()+"/v1/chat/completions", body)
	if err != nil {
		return benchRun{}, "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return benchRun{}, "", upstreamError(resp)
	}

	chat, err := translate.CollectChat(resp.Body, inst.Runtime().Model, 0, start)
	if err != nil {
		return benchRun{}, "", err
	}
	wall := time.Since(start)

	m := chat.Metrics
	run := benchRun{
		PromptTokens:  m.PromptEvalCount,
		PromptMS:      msOf(m.PromptEvalDuration),
		PromptTPS:     rate(m.PromptEvalCount, m.PromptEvalDuration),
		PredictTokens: m.EvalCount,
		PredictMS:     msOf(m.EvalDuration),
		PredictTPS:    rate(m.EvalCount, m.EvalDuration),
		WallMS:        msOf(wall),
		DoneReason:    chat.DoneReason,
	}
	// A reasoning model under ignore_eos never closes its think block, so the
	// whole generation is thinking and the content is empty.
	sample := chat.Message.Content
	if sample == "" {
		sample = chat.Message.Thinking
	}
	return run, truncate(sample, benchSampleChars), nil
}

// aggregate pools the runs rather than averaging their rates, so a short run
// does not weigh as much as a long one.
func aggregate(runs []benchRun) benchRun {
	var total benchRun
	for _, r := range runs {
		total.PromptTokens += r.PromptTokens
		total.PromptMS += r.PromptMS
		total.PredictTokens += r.PredictTokens
		total.PredictMS += r.PredictMS
		total.WallMS += r.WallMS
	}
	total.PromptTPS = ratePerMS(total.PromptTokens, total.PromptMS)
	total.PredictTPS = ratePerMS(total.PredictTokens, total.PredictMS)
	return total
}

// benchWords are short, common English words: one BPE token each in the
// vocabularies this drives, so a word count approximates a token count. The
// prompt token counts reported back are llama.cpp's own, not this estimate.
var benchWords = strings.Fields(`the of and to in a is that it for on with as at by from
	but not are was were be have has had do does did will would can could should may
	one two three four five six seven eight nine ten first last next time year day
	work part place way thing point line side form group number word name case fact
	over under after before while about against between through during without within`)

func fillerPrompt(words int, seed int64) string {
	rng := rand.New(rand.NewSource(seed))
	var b strings.Builder
	for i := 0; i < words; i++ {
		if i > 0 {
			b.WriteByte(' ')
		}
		b.WriteString(benchWords[rng.Intn(len(benchWords))])
	}
	return b.String()
}

func (s *Server) handleBenchStatus(w http.ResponseWriter, r *http.Request) {
	inst := s.Super.Current()
	if inst == nil {
		writeJSON(w, http.StatusOK, map[string]any{"loaded": false})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"loaded":     true,
		"runtime":    inst.Runtime(),
		"fit":        fitOf(inst.Fit()),
		"started_at": inst.StartedAt(),
		"expires_at": inst.ExpiresAt(),
		"uptime_ms":  msOf(time.Since(inst.StartedAt())),
	})
}

func fitOf(f supervisor.Fit) benchFit {
	return benchFit{
		OffloadedLayers: f.OffloadedLayers,
		TotalLayers:     f.TotalLayers,
		CPUBufferMiB:    f.CPUBufferMiB,
		KVBufferMiB:     f.KVBufferMiB,
		VRAMMiB:         f.VRAMMiB,
	}
}

func rate(tokens int, d time.Duration) float64 {
	if tokens <= 0 || d <= 0 {
		return 0
	}
	return float64(tokens) / d.Seconds()
}

func ratePerMS(tokens int, ms float64) float64 {
	if tokens <= 0 || ms <= 0 {
		return 0
	}
	return float64(tokens) * 1000 / ms
}

func msOf(d time.Duration) float64 {
	return float64(d) / float64(time.Millisecond)
}

func truncate(s string, n int) string {
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return string(r[:n]) + "…"
}

func ptr[T any](v T) *T { return &v }
