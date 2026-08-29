package supervisor

import (
	"strconv"

	"github.com/oswald/alpakka/internal/config"
)

// logVerbosity is the llama-server log threshold. It has to be high enough that
// the "offloaded N/M layers to GPU" line is printed, because that line is what
// the fit check reads. The output is consumed by the supervisor, not the user.
const logVerbosity = "5"

// Args builds the llama-server command line for a runtime.
func Args(rt config.Runtime, port int) []string {
	args := []string{
		"--model", rt.ModelPath,
		"--alias", rt.Model,
		"--host", "127.0.0.1",
		"--port", strconv.Itoa(port),
		"--no-webui",
		"-lv", logVerbosity,

		// Chat templating is llama.cpp's job. Its jinja path is the only place
		// chat_template_kwargs exists, and reasoning_effort rides on that.
		"--jinja",

		// Never let llama.cpp quietly shrink the context or the offload to make
		// something fit. Explicit settings must be honoured or the load must
		// fail; a silently downgraded load is the failure mode this project
		// exists to avoid.
		"--fit", "off",

		"-ngl", strconv.Itoa(rt.NumGPU),
		"--parallel", strconv.Itoa(rt.Parallel),
		"-c", strconv.Itoa(rt.NumCtx),
	}

	if rt.FlashAttn != "" {
		args = append(args, "--flash-attn", rt.FlashAttn)
	}
	if rt.CacheTypeK != "" {
		args = append(args, "--cache-type-k", rt.CacheTypeK)
	}
	if rt.CacheTypeV != "" {
		args = append(args, "--cache-type-v", rt.CacheTypeV)
	}

	if rt.SpecType != "" {
		args = append(args, "--spec-type", rt.SpecType)
		if rt.SpecDraftNMax > 0 {
			args = append(args, "--spec-draft-n-max", strconv.Itoa(rt.SpecDraftNMax))
		}
		if rt.SpecDraftNMin > 0 {
			args = append(args, "--spec-draft-n-min", strconv.Itoa(rt.SpecDraftNMin))
		}
	}

	// The vision projector reserves over a gigabyte of VRAM whether or not any
	// request uses it, which on a 16 GB card is often the difference between
	// fitting and spilling.
	if rt.ProjectorPath != "" {
		args = append(args, "--mmproj", rt.ProjectorPath)
	} else {
		args = append(args, "--no-mmproj")
	}

	return args
}
