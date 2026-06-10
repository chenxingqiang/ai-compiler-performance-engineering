# AI Systems Performance Engineering — Book Errata and Manuscript-vs-Code Review

This is the master index of errata found by reviewing the book text against the
repository code examples. Each erratum gives the exact text to SEARCH FOR in the
manuscript and the text to REPLACE IT WITH, so a fix can be applied directly.

Per-chapter detail lives in `chNN-fix.md`. This file is the index plus the
methodology, the manuscript-extraction notes, and the per-chapter review status.

## How the markdown manuscript was produced

The book PDF was converted to per-chapter markdown under `book/chNN.md` (plus
`book/preface.md` and `book/appendix.md`) with
`core/scripts/extract_book_from_pdf.py` (PyMuPDF). The extractor reconstructs
headings, body paragraphs (rejoined across page breaks, de-hyphenated using the
PDF's own U+2010-vs-U+002D encoding), fenced code listings (indentation
preserved), TIP/NOTE callouts (detected by the O'Reilly icon), figure/table
captions, and tables, and strips the running header/footer band.

Regenerate (the book text is copyrighted, so `book/` is git-ignored and stays
local):

```bash
pip install pymupdf
python core/scripts/extract_book_from_pdf.py \
  --pdf "/path/to/AI_Systems_Performance_Engineering.pdf" \
  --out book --front-matter
```

Fidelity check: for every chapter the set of `.cu/.py/.cuh` sample labels in
`book/chNN.md` exactly equals the set in the raw PDF text (zero missing, zero
added).

## Verification rule for an erratum

An erratum is only listed here once its BEFORE text is confirmed to appear
verbatim in the manuscript (so the search-and-replace is actually applicable).
A proposed change whose BEFORE text is NOT in the book is recorded as
"already-applied / not-applicable" instead, not as an outstanding erratum.

## Outstanding errata

### CH10-1 (Chapter 10, prose): block-scoped pipeline collectives are CTA-collective, not per-warp

Location: Chapter 10, intra-kernel pipelining section, in the paragraph that
introduces the CUDA Pipeline API producer/consumer calls (just before the
warp-specialized pipeline listing). Search anchor: `synchronize only the specific
warps`.

Why: the sentence claims the collectives synchronize "only the specific warps,"
but for a block-scoped `cuda::pipeline` these calls are CTA-collective (every
thread in the block must execute them in the same order). The book contradicts
itself later in the same chapter, which states the correct rule:
"producer_acquire(), producer_commit(), consumer_wait(), and consumer_release()
must occur in a consistent collective sequence across the participating CTA
threads. Do not place those collectives inside divergent warp-role branches ..."

SEARCH FOR (verbatim, the manuscript wraps these across lines):

```text
The key advantage of using the CUDA Pipeline API’s producer and consumer calls
(e.g., pipe.producer_acquire(), pipe.producer_commit(), pipe.consumer_wait(),
and pipe.consumer_release()) is that they synchronize only the specific warps or
stages that actually need to hand off data. This is in contrast to forcing every thread in
a block to wait.
```

REPLACE WITH:

```text
For a block-scoped pipeline, the producer and consumer calls
(pipe.producer_acquire(), pipe.producer_commit(), pipe.consumer_wait(), and
pipe.consumer_release()) are collective CTA operations. Every thread in the block
must execute these calls in the same order.
```

Full BEFORE/AFTER (including the following paragraph and the related code-pattern
guidance) is in `ch10-fix.md`. The "Code Pattern" block in `ch10-fix.md` is
illustrative of the unsafe-vs-safe pattern; the book's actual Chapter 10 listing
already uses the safe (CTA-uniform) form, so only the prose change above is a
literal manuscript edit.

Related, lower-confidence (same chapter): the earlier sentence "Concurrently,
other warps in the block invoke pipe.consumer_wait(). This stalls only the
threads dependent on the stage indexed by s = tile % STAGES." uses the same
per-warp framing for a block-collective call. Consider rewording "This stalls
only the threads dependent on the stage" to note that the call is CTA-collective
and the wait gates the block on that stage.

## Already-applied / not-applicable (recorded, no manuscript change)

### CH11: pipeline-collective legality

`ch11-fix.md` previously proposed a CTA-uniform-collectives change. The published
Chapter 11 already uses that safe pattern (its listings carry the comments
"Block-scoped collectives remain CTA-uniform" and "Block-scoped pipeline
collectives stay CTA-uniform."). The proposed BEFORE and AFTER text do not appear
in the book, so there is nothing to change. See `ch11-fix.md`.

## Tooling accuracy fixes applied in this pass

These address "make sure the existing scripts are accurate":

1. `core/scripts/audit_book_code_alignment.py` — `LABEL_RE` lookbehind bug. It was
   `(?<![\w/.-])`, which silently dropped every sample label written as a path or
   `./script.py` (for example `./add_sequential.py` in Chapter 6). Relaxed to
   `(?<![\w-])`. Samples checked went from 26 to 31 with no spurious matches.
2. `core/scripts/audit_book_code_alignment.py` — added orphaned-registry-entry
   detection. It reports registry entries whose label no longer appears in any
   `book/chNN.md` (8 found, listed below). Non-fatal.
3. `core/scripts/book_sample_registry.json` — registered two PyTorch library paths
   that the regex fix newly surfaced from a Chapter 14 torch.compile traceback
   (`torch/_dynamo/convert_frame.py`, `torch/nn/modules/module.py`) as
   `pseudocode`, matching the existing `fx_graph_runnable.py` / `outputcode.py`
   debug-artifact entries. Audit is now green (0 hard mismatches).

Orphaned registry entries (label not present in the current book; left in place,
flagged by the audit, for an author decision to prune or re-add the label):

- ch4: `after_overlap_ddp.py`, `after_reinit_comm.py`, `before_reinit_comm.py`
- ch10: `warp_specialized_cluster_pipeline.cu`, `warp_specialized_pipeline.cu`
- ch11: `warp_specialized_pipeline_multistream.cu`,
  `warp_specialized_kernel_two_pipelines_multistream.cu`,
  `warp_specialized_two_pipelines_multistream_driver.cu`

## Per-chapter review status

Extraction is complete for all chapters. The "errata" column tracks the
manuscript-vs-code accuracy review.

- ch01 Introduction: extracted. Errata review: pending.
- ch02 Hardware Overview: extracted (no code listings in this chapter). Errata review: pending.
- ch03 OS/Docker/Kubernetes: extracted. Errata review: pending.
- ch04 Distributed Networking: extracted. Errata review: pending.
- ch05 Storage I/O: extracted. Errata review: pending.
- ch06 GPU Architecture/CUDA: extracted. Errata review: pending.
- ch07 Memory Access Patterns: extracted. Errata review: pending.
- ch08 Occupancy/Warp/ILP: extracted. Errata review: pending.
- ch09 Kernel Efficiency/AI: extracted. Errata review: pending.
- ch10 Intra-Kernel Pipelining: extracted. Errata: CH10-1 verified (see ch10-fix.md).
- ch11 Inter-Kernel Pipelining: extracted. Errata: none outstanding (ch11-fix already applied in book).
- ch12 Dynamic/Device-Side Orchestration: extracted. Errata review: pending.
- ch13 Profiling/Scaling PyTorch: extracted. Errata review: pending.
- ch14 PyTorch Compiler/Triton/XLA: extracted. Errata review: pending.
- ch15 Multinode Inference: extracted (no code listings in this chapter). Errata review: pending.
- ch16 Inference at Scale: extracted. Errata review: pending.
- ch17 Disaggregated Prefill/Decode: extracted. Errata review: pending.
- ch18 Advanced Prefill-Decode/KV: extracted. Errata review: pending.
- ch19 Dynamic/Adaptive Inference: extracted. Errata review: pending.
- ch20 AI-Assisted Optimization: extracted. Errata review: pending.
