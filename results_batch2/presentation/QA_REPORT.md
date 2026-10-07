# QA report: research-story deck

**Automated check:** `python scripts/presentation/check_story_deck.py`. Full output: `qa_check_output.txt`.
**Result: ALL CHECKS PASSED (70/70).**
**Structural validation:** pptx validator, all validations passed.
**Visual check:** all 13 slides rendered (LibreOffice → PDF → images) and inspected for overflow, overlap and legibility.

## The ten validation items

| # | Check | Result | How |
|---|---|---|---|
| 1 | Every number matches the frozen reports | **PASS** | 54 displayed values are recomputed from `references/batch1/*` and the frozen-checkpoint extraction, formatted as on the slide, and found in the slide text. This includes all 26 appendix L2 values (3 s.f.). The Z4 and Z4-20K collapse times recomputed from frozen checkpoints equal the frozen table bit-for-bit. The conditioning ranges equal the frozen profile. |
| 2 | No paper number shown as our result | **PASS** | 4.64 × 10⁻⁴ appears only on slides 3 and 8. Both carry a "published" or "paper" label. Slide 3 says "Published value — not our result". Slide 8 states that the budgets differ. |
| 3 | No failed Batch-1 result labelled successful | **PASS** | No "successful / solved / outperform / best PINN / state of the art" anywhere, including notes. Slide 7 carries the explicit box "STATUS: NO VALIDATED OPTIMIZED PINN YET". |
| 4 | No Mode-2 result | **PASS** | No "Mode 2" in any slide text or notes. |
| 5 | No railway result presented as validated | **PASS** | Railway appears only on slide 10, labelled "future work — no railway results exist yet". The timeline puts railway at FUTURE. |
| 6 | No individual published method called novel | **PASS** | "novel" appears only in "Honesty about novelty" (slide-9 notes). Slides 9 and 13 state that the components are "benchmarked, not claimed". The solver strategy is labelled "RESEARCH HYPOTHESIS — UNDER INVESTIGATION … Not an established contribution". |
| 7 | Every graph has a clear scientific question | **PASS** | Slide 5: "How close does each stage get to the real vibration?". Slide 6: three panel questions. Slide 8: chart title "Does additional physics computation delay dynamic collapse?". The checker verifies that a question is present on every data-graph slide. |
| 8 | The experiment ladder is understandable without technical knowledge | **PASS (manual)** | Slide 4 has 8 rungs, each with one plain-language question and no equations. |
| 9 | Slide 9 clearly explains what Batch 2 is | **PASS (manual)** | Batch-2 pipeline; NOT "one PINN recipe for every PDE" / BUT "choose a solving strategy based on the physics and the compute budget"; the stated quote; the prior-art note; the hypothesis box; status. |
| 10 | The final message matches the evidence | **PASS** | The full final message is verbatim in the slide-10 notes; a condensed version is on the slide. Every element is backed by slides 5–8: representation (X3/X4), conditioning (8.4 × 10³ → O(1)), compute (1.2 → 8.1 cycles), no validated PINN. |

## Repository safety

| Item | Result |
|---|---|
| ROOT working tree | clean (`git status --porcelain` empty), before and after |
| ROOT HEAD | `6b31d79` before and after this task. Tree `73a9391…` is identical to the frozen commit `d31864a` recorded in `docs/ROOT_PROVENANCE.md`. |
| ROOT writes | none. Checkpoints were opened read-only for inference. All outputs passed `physref.safety.assert_safe_output(strict=True)` into `results_batch2/presentation/`. |
| `scripts/check_root_untouched.py` | reports `ROOT_MODIFIED: YES` **because of the checkout, not a modification**. It expects HEAD `d31864a` on branch `claude/tender-ramanujan-v3w90d` (the checkout of the session that wrote PINN-RP's provenance). In this container ROOT is checked out at `6b31d79` on `claude/pinn-beam-vibration-opt-a946fl`, already the case before this task started. Both commits have the same tree. Nothing was repaired or changed in ROOT. |
| PINN-RP tests | 157 passed, 1 failed. The failure is `test_root_is_unchanged_at_frozen_commit`, the same HEAD/branch expectation as above. |
| Training executed | **NO** |
