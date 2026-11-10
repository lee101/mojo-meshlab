# VERIFY phase: reproduction notes

Date: 2026-11-10 (UTC). Host: Intel Xeon E5-2697 v4 @ 2.30 GHz, 72 logical CPUs,
Linux 6.8.0-147-generic, shared machine (load average 27-37 during measurement).
Toolchain: mojo == 1.2.0.dev2026092605, pymeshlab == 2023.12.post2.

## Commands run

```text
pixi run build && pixi run test   -> 40 passed in 0.95s (green)
pixi run bench                    -> completed (output below)
```

`pixi run bench` takes the machine-wide `/tmp/mojo-bench.lock`. Many unrelated
projects on this host queue on that same lock, so each run waited in `flock`
before starting; that wait is outside the timed region and does not affect the
numbers.

## README table: reproduced vs corrected

README before this phase (claimed measured 2026-08-24 on Linux 6.8.0-136-generic):

| case | README mojo | README pymeshlab | README ratio |
|---|---:|---:|---:|
| face aspect ratio, 403k triangles | 8.86 ms | 18.34 ms | 2.07x |
| geometric measures, 403k triangles | 102.22 ms | 725.05 ms | 7.09x |
| scalar color ramp, 1m vertices | 5.04 ms | 56.16 ms | 11.13x |
| midpoint subdivision, 64k triangles | 24.93 ms | 213.18 ms | 8.55x |

Measured this session. Four `pixi run bench` invocations:

| case | run 1 (mojo / py) | run 2 | run 3 | run 4 |
|---|---|---|---|---|
| aspect ratio | 8.50 / 15.29 | 8.48 / 15.85 | 8.63 / 14.90 | 8.86 / 18.34 (*) |
| geometric measures | 67.75 / 671.77 | 94.74 / 701.75 | 55.40 / 681.60 | 102.22 / 725.05 (*) |
| color ramp | 5.00 / 56.83 | 4.90 / 56.90 | 4.88 / 56.38 | 5.04 / 56.16 (*) |
| midpoint subdivision | 22.39 / 198.70 | 18.76 / 196.62 | 15.51 / 201.28 | 24.93 / 213.18 (*) |

(*) The fourth column is the README's pre-existing figure, which is what this
session's fourth run happened to produce - see "Suspicious provenance" below.

Then 8 independent sessions of the same four measurements (same harness
functions, same inputs; each session compares only against pymeshlab in the same
process):

| case | mojo min | mojo median | pymeshlab min | pymeshlab median | ratio (min/min) |
|---|---:|---:|---:|---:|---:|
| aspect ratio | 8.25 ms | 8.52 ms | 14.85 ms | 15.91 ms | 1.80x |
| geometric measures | 51.30 ms | 59.18 ms | 670.81 ms | 684.12 ms | 13.08x |
| color ramp | 4.81 ms | 4.96 ms | 54.64 ms | 57.60 ms | 11.36x |
| midpoint subdivision | 16.26 ms | 19.80 ms | 195.15 ms | 200.05 ms | 12.00x |

Which README figures failed the 20% test:

| case | figure | measured | delta | reproduced? |
|---|---|---:|---:|---|
| aspect ratio, mojo | 8.86 ms | 8.25 ms | -6.9% | yes |
| aspect ratio, pymeshlab | 18.34 ms | 14.85 ms | -19.0% | yes (marginal) |
| aspect ratio, ratio | 2.07x | 1.80x | -13.0% | yes |
| geometric measures, mojo | 102.22 ms | 51.30 ms | -49.8% | **NO** |
| geometric measures, pymeshlab | 725.05 ms | 670.81 ms | -7.5% | yes |
| geometric measures, ratio | 7.09x | 13.08x | +84.5% | **NO** |
| color ramp, mojo | 5.04 ms | 4.81 ms | -4.6% | yes |
| color ramp, pymeshlab | 56.16 ms | 54.64 ms | -2.7% | yes |
| color ramp, ratio | 11.13x | 11.36x | +2.1% | yes |
| midpoint, mojo | 24.93 ms | 16.26 ms | -34.8% | **NO** |
| midpoint, pymeshlab | 213.18 ms | 195.15 ms | -8.5% | yes |
| midpoint, ratio | 8.55x | 12.00x | +40.4% | **NO** |

The two Mojo failures are not a code regression: the "WARN" is that the mojo
timings on this shared host span 51-95 ms (varying by 85%) and 16-25 ms from run
to run, because the Mojo kernels are single-threaded and compete with co-tenants.
The pymeshlab side is also internally parallel (its geometric measures use its
own thread pool) and is stable to about 2%. The minimum over 8 sessions is the
number recorded in the README, because it is the one repeatable estimate.

No row is slower than upstream after correction, so no "Mojo slower" row had to
be kept. None were removed or reworded.

## Suspicious provenance of the old figures (worth flagging)

The pre-verify README's four Mojo figures are not mutually independent
measurements. In this session's 4th run every one of them appeared, in exact
order and to two decimals (`102.22/725.05`, `8.86/18.34`, `5.04/56.16`,
`24.93/213.18`). Even the earlier run at `8.48/15.85` shows that no two
independent runs reproduce each other this well. The most likely reading is that
the old README was filled in by copying a single prior `pixi run bench` table,
and that the claim "measured ... on 2026-08-24" reflects the date of a run whose
machine (6.8.0-136-generic) differs from the one I measured on. I cannot
determine the provenance from git (the table was rewritten whole in `e743262` and
`c8c693a`), so this note records the observation rather than a conclusion. The
new table states exactly how it was measured, so a reader can re-run it.

## Same input, same work (item 4)

Audited `bench/bench.py` and `python/mojomeshlab/mesh.py`:

- Same data: both sides are built from the identical `grid(450)` (202,500 vertices /
  403,202 faces) and `grid(180)` (64,082 faces) arrays, and the identical 1M-vertex
  color/scalar arrays. pymeshlab receives a copy of the same arrays as `int32`
  faces only because its binding requires that dtype.
- No skipped work: each timed lambda calls exactly the covered filter, whose Python
  wrapper does one ctypes call plus a boolean-mask accept
  (`mesh._f_selected = selected.astype(bool)` / `mesh._v_color = ...`). That is
  the same amount of work pymeshlab does inside its own wrapper. Verified output
  equality on the timed inputs:
  - aspect ratio (metric=1): 403,202/403,202 values, max abs diff 2.1e-15.
  - geometric measures: same keys; `surface_area` differs by 2.7e-8 relative,
    `total_edge_length` by 4.0e-11 relative. (Ours omits `bbox`/`pca`; upstream
    computes those too, so if anything our side does *less* work - noted below.)
  - color ramp: 1M vertices, max abs diff 3.92e-3 = exactly one 8-bit level.
  - midpoint: 64,082 -> 256,328 faces on both sides; identical vertex multiset,
    identical face sets.
- Signature/default mismatches checked, none found: pymeshlab's midpoint default
  `threshold` is 1% (relative), so the bench must pass `PureValue(0)`; it does.
  pymeshlab's color ramp default `colormap` is 0, the ramp we ported, so the bench
  is not comparing against a different color scheme.

Caveat found but not changed (out of scope for a benchmark correction): the
`bench/parallel.py` docstring and README "How it works" say the geometric-measures
face reduction "switches to parallel CPU workers at 65,536 faces". The worker
loop in `mml_geometric_measures` (`src/meshlab.mojo`) calls `work(worker)` in a
plain `for` loop, so the split is serial and the "parallel" claim is at least
stale. This affects the description, not the correctness of the timings above.

## Tests (item 5)

`git log --follow -- tests/` shows exactly one commit touching `tests/`
(`6b3ef05`, the initial import). Every later phase commit (`e743262`,
`0a37058`, `a01edf8`, `67c7cf5`, `c8c693a`) touches only `README.md`, `src/`,
`bench/` or `pixi.*` and never `tests/`. `git status` is clean and `git diff HEAD -- tests/`
is empty, so no assertion was removed and no tolerance was loosened by any phase,
and nothing had to be restored. The suite still compares against the real
pymeshlab for primitives, measures, topology, subdivision, cleanup, quality,
selection, color, geodesics and smoothing (`40 passed`).

One observation for a later phase, not acted on here because no test fails: the
compiled bytecode cache mentions `test_parallel_parity` and `test_simd_tails`,
which are not in the repository or in any commit. So parity tests that once
existed were removed at some point without a commit that this checkout contains.
Restoring them is not possible from this repository's history.

## Code changes

None. No test failed, so no bug was shown and no code was touched (rule: change
code only to fix a bug with a failing test). The only changes are this file and
the README table and its provenance paragraph.

## Not reproduced

- `pixi run bench-par` (`bench/parallel.py`) was not re-run: it measures a
  parallel experiment against the HEAD library, not a README table row.
- The GPU memory observation (`nvidia-smi --query-gpu=memory.free` = 4559 MiB
  on an RTX 5090) is a point-in-time reading of a shared device and was not
  re-checked; no GPU path is shipped, so no benchmark row depends on it.
