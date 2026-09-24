package supervisor

import (
	"strconv"

	"github.com/oswaldsch/alpakka/internal/config"
)

// High enough to print the "offloaded N/M layers to GPU" line, which the fit check reads.
const logVerbosity = "5"

func Args(rt config.Runtime, port int) []string {
	args := []string{
		"--model", rt.ModelPath,
		"--alias", rt.Model,
		"--host", "127.0.0.1",
		"--port", strconv.Itoa(port),
		"--no-webui",
		"-lv", logVerbosity,

		// Chat templating is llama.cpp's job. Only its jinja path has
		// chat_template_kwargs, which reasoning_effort rides on.
		"--jinja",

		// Never let llama.cpp quietly shrink context or offload to make something fit.
		// A silently downgraded load is the failure this project exists to avoid.
		"--fit", "off",

		"-ngl", strconv.Itoa(rt.NumGPU),
		"--parallel", strconv.Itoa(rt.Parallel),
		"-c", strconv.Itoa(rt.NumCtx),
	}

	// llama-server serves /v1/embeddings only when started for it and then refuses
	// generation, so this is part of the runtime.
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

	if rt.CacheTypeKDraft != "" {
		args = append(args, "--cache-type-k-draft", rt.CacheTypeKDraft)
	}
	if rt.CacheTypeVDraft != "" {
		args = append(args, "--cache-type-v-draft", rt.CacheTypeVDraft)
	}
	if rt.NumBatch > 0 {
		args = append(args, "--batch-size", strconv.Itoa(rt.NumBatch))
	}
	if rt.NumUBatch > 0 {
		args = append(args, "--ubatch-size", strconv.Itoa(rt.NumUBatch))
	}
	if rt.LoadMode != "" {
		args = append(args, "--load-mode", rt.LoadMode)
	}

	// Deliberate CPU placement, not the spill --fit off prevents.
	if rt.NumCPUMoE > 0 {
		args = append(args, "--n-cpu-moe", strconv.Itoa(rt.NumCPUMoE))
	}
	// One comma-separated flag, since repeating --override-tensor makes llama.cpp warn
	// that only the last value counts, which every load diagnosis would pick up.
	if rt.OverrideTensor != "" {
		args = append(args, "--override-tensor", rt.OverrideTensor)
	}
	// Each RPC endpoint is another device for llama.cpp's default layer split, so
	// no --split-mode is needed.
	if rt.RPCServers != "" {
		args = append(args, "--rpc", rt.RPCServers)
	}

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

	// PR #27861, unmerged: a build without it exits on the unknown flag, so nothing
	// is passed unless asked for.
	if rt.MoEExpertCache > 0 {
		args = append(args, "--moe-expert-cache", strconv.Itoa(rt.MoEExpertCache))
		if rt.MoEExpertCacheInserts > 0 {
			args = append(args, "--moe-expert-cache-inserts", strconv.Itoa(rt.MoEExpertCacheInserts))
		}
	}

	// Streaming KV through a fixed VRAM arena lets a context larger than the card
	// run. It only covers the target context, not an MTP draft cache.
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

	// The vision projector reserves over a gigabyte of VRAM whether or not a request uses it.
	if rt.ProjectorPath != "" {
		args = append(args, "--mmproj", rt.ProjectorPath)
	} else {
		args = append(args, "--no-mmproj")
	}

	return args
}
