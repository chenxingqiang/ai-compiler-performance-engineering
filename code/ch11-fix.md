# Chapter 11 Fixes

## Status: no outstanding change (already correct in the published book)

An earlier draft of this file proposed making the block-scoped pipeline
collectives CTA-uniform and warned against placing them only inside warp-role
branches. The published manuscript already does exactly that, so there is nothing
left to search-and-replace in Chapter 11.

The Chapter 11 listings issue `producer_acquire` / `producer_commit` /
`consumer_wait` / `consumer_release` uniformly across the CTA and keep
role-specific work between `cta.sync()` points. The code even states the rule in
inline comments:

- "Block-scoped collectives remain CTA-uniform" (the two-pipeline ping-pong listing)
- "Block-scoped pipeline collectives stay CTA-uniform." (the cluster + CUDA-streams listing)

Verification against the book text (`book/ch11.md`, extracted from the PDF):

- The proposed BEFORE prose ("One subtle but important legality rule ...", "all CTA
  threads participate") does not appear anywhere in Chapter 11.
- The proposed AFTER sentence ("... mismatched collective sequence can deadlock")
  is likewise absent, because the safe pattern is already the one printed.

The live correctness erratum on this topic is in Chapter 10 prose, not Chapter 11.
See `ch10-fix.md` and `BOOK-ERRATA.md` (CH10-1).
