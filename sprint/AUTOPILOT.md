# AUTOPILOT — unattended research loop (authorised by the PI in chat, 2026-10-09 00:52 IST)

A scheduled task starts a fresh Claude Code session every 2 hours. Each session follows this file,
does as much as fits, commits and pushes, updates sprint/STATUS.md, and exits. Sessions share no
memory: sprint/STATUS.md is the only state. Read it first, write it last.

## Mission

Find, validate and write up a genuinely novel, demonstrably better method for the full-field damped
Euler-Bernoulli beam benchmark (B1), as defined in sprint/SPRINT_24H_v2_CLAUDE_CODE.md:

SUCCESS = a method that, against B1 at the same 640,000-PDE-evaluation budget on THIS hardware,
  (a) lowers |fitted decay - 3.54| AND raises persistence (with amplitude ratio at t = 0.5 and 1.0 s
      within [0.5, 2]) AND lowers L2_exact, in 3/3 seeds (1234, 1235, 1236);
  (b) costs at most 1.5x B1 wall-clock per PDE evaluation;
  (c) passes a prior-art table: >= 5 closest papers (opened, not snippets), each with what it does,
      and a precise statement of the mathematical/algorithmic object Q that none of them has.
When SUCCESS is met, or on 2026-10-12 23:59 IST, write sprint/AUTOPILOT_REPORT.md, set STATUS to DONE,
and do nothing in later sessions.

## Session protocol

1. `git fetch origin`; check out branch b2-e05-window-law and pull. Also fetch every `claude/*` branch;
   if one has commits not yet in b2-e05-window-law (work from other sessions), merge it first and note it in STATUS.md.
2. Read sprint/STATUS.md. If `state: DONE`, exit immediately.
   If `lock_until` is in the future, another session is running: exit immediately.
   Otherwise set `lock_until` = now + 110 min, commit, push.
3. Setup: `nproc; pip install -r requirements.txt; python -m pytest tests/ -q`.
4. Do the next step(s) of the phase recorded in STATUS (below). Run training in the background,
   up to (cores - 1, at least 1) runs in parallel, 1 thread each; poll; never let a single command
   block for more than ~5 minutes. Stop launching new runs once ~80 min of the session have passed;
   wait for running ones; never leave a run unfinished at exit (kill and record it as "interrupted").
5. Commit and push after every finished run and at exit: code, docs, results_batch2/ outputs,
   STATUS.md (phase, what was done, numbers, next step, lock_until cleared).
   If pushing to b2-e05-window-law is refused, push to a `claude/autopilot-...` branch and record its
   name in STATUS.md there.

## Phases

A. Block 1 of the sprint brief: docs/hypotheses/B2-E05.md (before any training), src/physref/windowed.py,
   configs/batch2/B2-E05_window_law.yaml, dispatch in experiments/run_batch2.py, unit tests
   (IC/BC exact, u and u_t continuous at every hand-off, W1000 reproduces B1).
B. Blocks 2-3: seed 1234 at W1000, W050, W100, W250 (in that order). Compare measured excess decay with
   the prediction from measured R per window. Apply the gate in the brief and record PASS/PARTIAL/FAIL.
C. If PASS or PARTIAL: Blocks 4-6 (seeds 1235/1236, Pareto, window-selection rule). Then check SUCCESS.
D. If FAIL, or C does not reach SUCCESS: method search, at most 3 candidates per session, 10 in total.
   For each candidate:
   1. Literature: WebSearch/WebFetch for 2025-2026 work on PINN long-horizon oscillation, artificial
      damping, time-marching, temporal gradient starvation, separable/causal PINNs. Open the papers.
   2. Ground the idea in the evidence already in the repo: B2-E04 (residual ~88% in mode 1; late-window
      spatial gradients 28-45x temporal), tonight's exploratory runs (sprint/exploratory_runs_*.csv:
      loss amplitude-scaling flips the bias; energy penalties collapse to static fields).
   3. Pre-register in docs/hypotheses/B2-E0N.md (hypothesis, falsifier, metrics) BEFORE running.
   4. Add its IDs to configs/batch2/APPROVED_EXPERIMENTS.txt with a comment citing this file
      (pre-authorised families: B2-E06 ... B2-E15; at most 60 runs in total; 640k evaluations each
      unless the pre-registration justifies otherwise).
   5. Seed 1234 first; promote to 3 seeds only if it beats B1 on decay error and persistence at seed 1234.
   A candidate may combine existing published mechanisms only if the combination is motivated by a
   specific measured failure and the novelty claim is about that mechanism, never "we combined X+Y".
E. Write-up when SUCCESS (or at the deadline): update docs/papers/PAPER1_DRAFT.md (create it) with
   method, theory, prior-art table, all results including failures, limitations; and
   sprint/AUTOPILOT_REPORT.md with a one-page summary for the PI.

## Non-negotiable rules

- Never edit frozen files (configs/frozen, src/beampinn) or anything under ROOT; outputs only under results_batch2/.
- Same B1 configuration for comparisons: CPU, float32, 1 thread per run, same seeds and budget.
- Every results row: L2_exact, L2_late_exact, persistence (displacement and velocity), amplitude ratio at
  t = 0.1/0.25/0.5/0.75/1.0 s, fitted frequency and decay, PDE_residual_rel, train_seconds, pde_evaluations,
  peak_rss_mb. Persistence alone cannot detect growth.
- Log every configuration tried, including failures, in sprint/RUN_LOG.csv. No silent tuning; no
  selective reporting; no claim of novelty without the prior-art table; no claim of success outside the
  SUCCESS definition. If unsure, say so in STATUS.md.
- Do not delete or rewrite earlier results or history.
