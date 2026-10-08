# STATUS (autopilot state; read first, write last)

state: RUNNING
phase: A
lock_until: 2026-10-09 02:45 IST
deadline: 2026-10-12 23:59 IST
candidates_tried: 0
runs_used: 0

## PI note (2026-10-09 00:58 IST) — READ BEFORE CONTINUING PHASE A
Block 1 was completed in a separate browser session and is now merged into this branch (merge commit
on top of 9321298; source commit 55cd372 on claude/zen-ptolemy-iuhtyj):
docs/hypotheses/B2-E05.md, src/physref/windowed.py, configs/batch2/B2-E05_window_law.yaml,
`kind: windowed` dispatch in experiments/run_batch2.py, tests/test_b2_windowed.py (17 tests pass,
including W1000 bit-identical to the B1 Trainer over 30 steps).
Instructions: USE THIS implementation. If you already wrote a parallel Block-1 implementation, do not
keep two: discard yours unless it is demonstrably better and equally tested, and record the decision here.
Before training, re-run the FULL test suite (`python -m pytest tests/ -q`), because experiments/run_batch2.py
changed after the last full run. If it passes, Phase A is complete: move to Phase B.
The B2-E05 IDs for seeds 1234-1236 are already approved in configs/batch2/APPROVED_EXPERIMENTS.txt.
Smoke test (W050, 100 steps, scratch only): 0.17 s/step, ~1.5 s per window for R_k, 85 s stitched evaluation,
peak ~1.4 GB.

## Last session
(none yet)

## Next step
Phase A: Block 1 of sprint/SPRINT_24H_v2_CLAUDE_CODE.md (hypotheses doc, windowed trainer, tests).
