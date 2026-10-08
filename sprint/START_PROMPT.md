# Start prompt for Claude Code (browser) — B2-E05 window-law sprint

Paste into Claude Code on this repo, branch `b2-e05-window-law`:

> Follow sprint/START_PROMPT.md exactly.

---

## Instructions for Claude Code

Read sprint/SPRINT_24H_v2_CLAUDE_CODE.md fully, then README.md, docs/BATCH2_READY_FOR_TRAINING.md,
src/beampinn/training/trainer.py, src/beampinn/models/constraints.py, src/physref/arms.py,
src/physref/persistence.py, experiments/run_batch2.py and scripts/analyze_b2_e04.py (ff_mode_stable).
sprint/homogeneity_prototype.py is a working example of subclassing Trainer without touching frozen
code; sprint/exploratory_runs_2026-10-09.csv holds tonight's exploratory numbers.

Goal: Blocks 1-3 of the brief; stop at the Block-3 decision gate and report.

Environment
- First run: `nproc; python --version; pip install -r requirements.txt; python -m pytest tests/ -q`.
  Report the core count.
- Work on branch b2-e05-window-law. Commit and push after every block AND after every finished run
  (code, docs, results_batch2/ outputs) so nothing is lost if the session ends.
- Run training in the background (`nohup ... &`), as many arms in parallel as cores minus one (at least
  one), and poll the logs; never block a single command for more than a few minutes.
  With one core, run in the order W1000 -> W050 -> W100 -> W250.

Hard rules
- Never edit frozen files (configs/frozen, src/beampinn) or anything under ROOT. New code:
  src/physref/windowed.py; tests in tests/; spec configs/batch2/B2-E05_window_law.yaml. Extending
  experiments/run_batch2.py to dispatch the new arm is allowed if existing arms behave identically.
- Write docs/hypotheses/B2-E05.md (H-E05a-d and falsifiers from the brief) BEFORE any training.
- Approval gate: do NOT add IDs to configs/batch2/APPROVED_EXPERIMENTS.txt on your own. When the
  hypotheses doc and tests are ready, list the exact IDs and wait. Add them only after the user replies
  with those IDs, and commit with the message "PI approval: <IDs>".
- Same configuration as B1: CPU, float32, 1 thread per run, seed handling, 640,000 PDE evaluations total.
- Window hand-off: u_k = a_k(x) + tau*b_k(x) + tanh^2(w1*tau)*Phi(x)*A0*N_k(x,tau), with a_k, b_k the
  previous window's u and u_t at t_k projected onto the first 8 fixed-fixed modes. Unit-test: IC/BC
  exact; continuity of u and u_t across hand-offs; the 1-window arm (W1000) reproduces logged B1.
- Hardware differs from the logged runs. If W1000 does not match logged B2-E01 L2 (0.52639, seed 1234)
  to 4 digits, report the difference plainly and use THIS machine's W1000 as the reference.
- Every results row includes amplitude ratio R(t) at t = 0.1, 0.25, 0.5, 0.75, 1.0 s (persistence
  cannot detect growth), fitted decay, and measured unresolved residual R per window.
- Report failures as plainly as successes. Log every setting tried; no silent tuning.

Start by showing the implementation plan before writing code.
