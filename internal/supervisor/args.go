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

	// llama-server answers /v1/embeddings with "This server does not support
	// embeddings" unless it was started for it, and refuses generation when it
	// was. There is no process that does both, which is why this is part of the
	// runtime rather than a per-request flag.
	if rt.Embedding {
		args = append(args, "--embeddings")
	}
	if rt.Pooling != "" {
		args = append(args, "--pooling", rt.Pooling)
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

	// Deliberate CPU placement, not the spill --fit off exists to prevent: the
	// offload line these do not move is still required to read N/N.
	if rt.NumCPUMoE > 0 {
		args = append(args, "--n-cpu-moe", strconv.Itoa(rt.NumCPUMoE))
	}
	// One comma-separated flag, not one flag per pattern: llama.cpp still appends
	// a repeated --override-tensor, but its parser warns that it is deprecated
	// and that only the last value counts, and that warning is picked up as a
	// notable line in every load diagnosis.
	if rt.OverrideTensor != "" {
		args = append(args, "--override-tensor", rt.OverrideTensor)
	}
	// Each RPC endpoint is another device the layer split can land on, so this
	// needs no --split-mode: llama.cpp already splits by layer across whatever
	// devices are present.
	if rt.RPCServers != "" {
		args = append(args, "--rpc", rt.RPCServers)
	}

	// Unset passes nothing, so llama.cpp keeps using every device it sees with
	// its default layer split.
	if rt.Device != "" {
		args = append(args, "--device", rt.Device)
	}
	if rt.TensorSplit != "" {
		args = append(args, "--tensor-split", rt.TensorSplit)
	}
	if rt.SplitMode != "" {
		args = append(args, "--split-mode", rt.SplitMode)
	}
	if rt.MainGPU >= 0 {
		args = append(args, "--main-gpu", strconv.Itoa(rt.MainGPU))
	}
	if rt.NoKVOffload {
		args = append(args, "--no-kv-offload")
	}

	// PR #27861, unmerged: a build without it exits on the unknown flag rather
	// than ignoring it, which is why nothing is passed unless asked for.
	if rt.MoEExpertCache > 0 {
		args = append(args, "--moe-expert-cache", strconv.Itoa(rt.MoEExpertCache))
		if rt.MoEExpertCacheInserts > 0 {
			args = append(args, "--moe-expert-cache-inserts", strconv.Itoa(rt.MoEExpertCacheInserts))
		}
	}

	// Streaming the KV cache through a fixed VRAM arena is what lets a context
	// larger than the card run at all. It only covers the target context: an MTP
	// draft cache is not streamed and still needs its own VRAM.
	if rt.KVStreamArenaMiB > 0 {
		args = append(args, "--kv-stream-arena-mib", strconv.Itoa(rt.KVStreamArenaMiB))
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
