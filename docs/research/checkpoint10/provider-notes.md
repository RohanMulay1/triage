# Checkpoint 10: hosted-provider recovery

The documentation review covers hosted text-generation integration, model
identity, request/response options, reasoning controls, model lifecycle, and Groq
compatibility and pricing. It is not a claim to have read all NVIDIA developer
products (CUDA, graphics, self-hosted deployments, etc.). These are the sources
that affect this API-only repository:

- https://docs.api.nvidia.com/nim/docs/introduction
- https://docs.api.nvidia.com/nim/reference/llm-apis
- https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-3-nano-30b-a3b-infer
- https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-3-super-120b-a12b-infer
- https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-3-ultra-550b-a55b-infer
- https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-3-super-120b-a12b
- https://docs.nvidia.com/rag/latest/enable-nemotron-thinking.html
- https://console.groq.com/docs/openai
- https://console.groq.com/docs/deprecations
- https://console.groq.com/docs/models
- https://console.groq.com/docs/rate-limits

Reviewed 2026-09-08. Some NVIDIA index links redirect to the documentation home
page; several old model cards remain published after hosted retirement. The
model catalog is not a liveness guarantee. Direct capped generation is required.

## Observations and repair

The authenticated NVIDIA catalog returned a Nano alias that failed with 404.
The documented canonical Nano ID returned 410 (retired September 1). Super
returned 503 on three probes. These responses are preserved in separate files;
we do not rewrite a failed probe after discovering a working model.

Lightning 3.5 and Ultra both generated real completions. Their new research IDs
are `nim-nemotron-lightning-30b` and `nim-nemotron-ultra-550b`. The adapter sends
explicit non-streaming settings, disables thinking for this bounded-answer
experiment, and does not request unverified logprob support. Request settings
are captured in model snapshots. The historical Llama IDs and router default
are unchanged. This is a new model pair, not the originally planned 8B/70B test.

Groq also retired the original Llama pair for free/developer accounts. The
supplied key accesses the catalog and both GPT-OSS 20B and 120B generate real
completions. Groq does not support logprob parameters; the old adapter flag was
wrong and is now false. `groq-gpt-oss-20b` and `groq-gpt-oss-120b` explicitly pin
provider and low reasoning effort. Listed rates are snapshotted; they are not
assumed free merely because a key has free-tier access.

Fallback is selected for a fresh run by explicit model IDs. We do not swap
providers or model families midway through an experimental run. If NVIDIA fails,
retain its failures and use the Groq pair in a separate fresh run, subject to the
remaining cap and the unchanged whole-run admission estimate. The current
conservative estimate for 120 Groq items exceeds the approved cap; it must refuse
rather than silently spend against a per-item cap. Model probes use a small
explicit completion limit and preserve raw response bodies and token counts.

## Scientific limits

A successful OK response establishes connectivity, not quality, repairability,
calibration, or a contribution. Missing logprobs change the signal mix and must be
disclosed. Non-thinking Nemotron and low-effort GPT-OSS are explicit experimental
settings, not vendor best-quality claims. Trial listed prices do not verify an
invoice. There is no cross-domain result or controller. Gate 1 remains
provisionally NARROW with the unread Yin & Zhang paper disclosed.

## Scoring envelope compatibility

The first Groq smoke used comparator version 1 and produced genuine provider
answers, but its scores were JSON objects enclosed in Markdown fences. Version 1
refused those responses before fan-out. Version 2 accepts only raw JSON or exactly
one whole `json` fence. It never extracts objects from explanatory prose or repairs
invalid JSON or values. Numeric, finite-range and exact-action-coverage validation
still runs after envelope decoding. Tests include a fenced out-of-range score.
The prompt and utility equation did not change; the versioned scoring action and
policy distinguish this serialization change from the preserved failed smoke.
