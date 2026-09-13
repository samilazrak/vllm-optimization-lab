# Interpretation

*[Version française](findings.fr.md)*

This document deliberately separates three things: what this campaign measured,
what the book measured, and what was not reproduced at all.

Every figure quoted here comes from [`results/summary.csv`](../results/summary.csv)
and the logs in [`results/raw/`](../results/raw/), produced on a 46 GB L40S with
vLLM 0.29.0.

## 1. The memory profile reproduces to within 1 %

| Measurement | This repo | Book | Gap |
|---|---|---|---|
| `base` weights | 27.52 GiB | 27.5185 GiB | identical |
| `base` KV cache | 11.06 GiB | 11.00 GiB | +0.5 % |
| `base` cacheable tokens | 72,496 | 72,064 | +0.6 % |
| `base` max concurrency | 1.77× | 1.76× | +0.6 % |
| `awq` weights | 9.44 GiB | 9.36 GiB | +0.9 % |
| `awq` KV cache | 29.09 GiB | 29.15 GiB | -0.2 % |
| `awq` cacheable tokens | 190,672 | 191,056 | -0.2 % |

The vLLM version used here is substantially newer than the book's, and yet
nothing moves. The reason is mechanical: these numbers are a subtraction between
the card's capacity and the model's size. The server only intervenes through the
margin it reserves for activations and CUDA graphs, and that margin has barely
changed between the two versions.

This is good methodological news: **the memory profile is the most stable and
most transferable part of the chapter.** It can be computed in advance and
checked by reading four log lines, without running a single benchmark.

On the serving side, reproduction is good on throughput (+1 % on both `base`
runs, +4 % on `awq__sharegpt`) and on ITL (42.4 ms against 43.2 on
`base__sharegpt`), which places model execution at the same level.

**One metric diverges sharply: TTFT.**

| Run | This repo | Book | Gap |
|---|---|---|---|
| `base__sharegpt` | 145.2 ms | 104.2 ms | +39 % |
| `awq__sharegpt` | 75.6 ms | 59.3 ms | +27 % |

The gap is too wide to ignore and too isolated to blame on the GPU: if
computation were slower, ITL and throughput would show it too. It points to the
request admission path, the tokenizer or the host rather than to decoding. That
part was not instrumented during the campaign, so the cause remains unknown. It
is recorded here as an unresolved divergence, not as a detail.

## 2. The AWQ gain is real, its explanation is not

The book lays out this causal chain: 4-bit weights free VRAM, the freed VRAM
goes to the KV cache, a larger cache allows bigger batches and fewer evictions,
so throughput rises. The first two links are confirmed by the table above. **The
third does not hold under these measurement conditions.**

The calculation that shows it is a single division. On ShareGPT, a request
averages 429 tokens (223 input, 206 output). The `base` KV cache, the smallest
of the three, holds 72,496 of them, which is room for **169 concurrent
requests**. The benchmark launches 10. The cache is used at 6 % of its usable
capacity, and at 9 % on Prefix Repetition.

In other words, `base` and `awq` both held all 10 in-flight requests without the
slightest memory pressure. Batching therefore cannot explain the throughput gap.
What does explain it:

| | `base` | `awq` | ratio |
|---|---|---|---|
| Model weights | 27.52 GiB | 9.44 GiB | **2.92×** |
| Mean ITL | 42.4 ms | 15.2 ms | **2.79×** |
| Output TPS (ShareGPT) | 230.8 | 640.1 | **2.77×** |
| Run duration | 1,783 s | 643 s | **2.77×** |

At low batch sizes, decoding is bound by memory bandwidth: for every token
generated, the GPU re-reads the entire set of weights. Dividing the weights by
2.92 therefore divides per-token time by a similar factor, and that is exactly
what we observe (2.79). The three other ratios follow at 2.77, which is the
signature of a system where **only one quantity changed**.

The distinction matters, because it changes the prediction the measurement
supports. If the gain came from batching, it would grow with concurrency. If it
comes from weight bandwidth, it is already fully present at a single request and
**erodes** as the batch grows, since the cost of reading the weights is then
amortized across several sequences.

This is not an error in the book: the cache expansion from 1.77× to 4.66×
sustainable concurrency is entirely real, and it is what matters on a saturated
workload or with long contexts. It is the measurement protocol that never
reaches it, by capping concurrency at 10. The correct conclusion is narrower
than the stated one: **on this workload, AWQ accelerates decoding; the memory
benefit stays in reserve.**

Testing it would mean re-running the matrix with `--max-concurrency` well past
170 on ShareGPT, and seeing whether the `awq` / `base` ratio collapses or holds.
That is the natural follow-up to this lab.

## 3. Manual tuning has no measurable effect

| Run | `awq` | `tuned` | Gap |
|---|---|---|---|
| ShareGPT | 1,334.2 TPS | 1,335.3 TPS | +0.08 % |
| Prefix Repetition | 2,873.0 TPS | 2,867.1 TPS | -0.2 % |

`tuned` nonetheless adds six flags: `--gpu-memory-utilization 0.95`,
`--enable-prefix-caching`, `--enable-chunked-prefill`, `--max-num-seqs 512`,
`--max-num-batched-tokens 8192` and `--block-size 16`. Two causes compound to
cancel their effect.

**Prefix caching and chunked prefill have been on by default since vLLM 0.29.**
Requesting them explicitly changes nothing: `base` was already benefiting from
them, which the reuse rate in point 4 confirms directly. The book presents them
as optimizations to switch on; they are now defaults, and part of the gain the
chapter attributes to tuning is already in the baseline.

**The only real gain is 0.94 GiB of KV cache** (29.09 to 30.03 GiB), obtained by
pushing `--gpu-memory-utilization` from 0.90 to 0.95. It applies to the resource
that point 2 shows was not binding. Widening a reserve already used at 6 %
produces nothing.

The result is negative, and it is the most useful of the set: it is a reminder
that a flag only improves a workload if it relaxes the constraint limiting it,
and that the constraint therefore has to be identified before anything gets
tuned.

## 4. Prefix caching worked without showing

The server logs a cumulative reuse rate. On the `base` run, which chains
ShareGPT then Prefix Repetition without restarting:

| Moment | Cumulative rate |
|---|---|
| End of the ShareGPT phase | 0.2 % |
| End of the Prefix Repetition phase | 49.5 % |

The two phases total 958,647 input tokens, 512,028 of them in the prefix phase.
Since the ShareGPT phase contributes almost no hits, the cumulative 49.5 %
concentrates on the second, which puts it **around 90 % reuse**. The mechanism
worked perfectly, and the 0.2 % against 90 % contrast is a good illustration of
what the choice of dataset decides: real conversational traffic shares almost
nothing, synthetic traffic with 10 prefixes shares almost everything.

And yet TTFT does not move: 150 ms on `base__prefix` with 512 input tokens,
against 145 ms on `base__sharegpt` with 223. Serving twice the prompt at the
same price is indeed the signature of the cache, but the absolute gain is
drowned. Prefill costs 145 ms when decoding a response costs roughly 8,600.

**Prefix caching pays off in TTFT, and this campaign measures throughput.** On an
interactive workload with short responses, the hierarchy would flip.

## 5. Distributed serving (not reproduced)

This section reports the book's results. **No measurement in this repository
confirms or contradicts them**: step 8 requires two multi-GPU pods with
different interconnects, which is outside this campaign's budget. The `tp2` and
`tp4` configs exist in `scripts/01-serve.sh` for anyone with the hardware.

**Result A: multi-GPU can be worse than single-GPU.**

| Instance | GPU | Interconnect | Winner |
|---|---|---|---|
| g6e.12xlarge | 4× L40S | PCIe, no NVLink | **single-GPU** |
| p4d.24xlarge | 8× A100 | NVLink | 4-GPU |

The L40S is nonetheless better than the A100 at single-GPU inference. The
interconnect is what decides: without NVLink, tensor parallelism's
synchronization overhead cancels the parallelism gain.

**Result B: even when distributed wins, four independent instances beat it on
throughput**: 9,816 against 3,926 TPS on p4d.

Hence the conclusion: **the benefit of going distributed is not throughput, it is
latency and the model size you can reach.** On p4d, going from 1 to 4 GPUs drops
TTFT from 66 to 33 ms. Horizontal scaling never reaches that gain: adding
replicas does not improve the latency of a single request.

## What the lab does not measure

- **The KV cache under pressure.** This campaign's main limitation, detailed in
  point 2: `--max-concurrency 10` leaves the cache at 6 % of its usable capacity,
  so the chapter's central lever is never exercised.
- **Quantized model degradation.** The harness measures throughput, not output
  quality at 4 bits. A 2.77× throughput gain says nothing about the precision
  cost, and the memory-versus-quality tradeoff is the second of the chapter's
  five tradeoffs.
- **The targeted techniques of step 7.** LMCache for prefill-heavy workloads and
  speculative decoding for decode-heavy ones each require their own protocol,
  with a dataset that leans firmly one way. The `tuned` config only covers the
  general caching and batching levers.
- **Variance between runs.** Every data point is a single run. The percent-level
  gaps quoted here, in particular the +1 % and +4 % against the book, are not
  distinguishable from noise without repetition.

## The guiding principle

> In practice, we spend most of our effort identifying which optimization
> techniques to apply rather than chasing the perfect configuration.

This campaign is an unintentional illustration of it. The two most instructive
results are a negative one (the `tuned` configuration adds nothing) and an
explanation that collapses under verification (the AWQ gain does not come from
batching). In both cases, what was missing was not a flag but the identification
of the constraint that actually limited the workload.

The resulting order of work: understand the scenario, pick a representative
dataset and representative metrics, apply the general optimizations to get a
solid baseline, and only then target the workload. A configuration overfitted to
one GPU and one traffic pattern is not portable, and may be bad elsewhere.
