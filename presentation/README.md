# Supervisor progress deck

`PINN_Research_Progress_Supervisor.pptx` (14 slides, 16:9) and `PINN_Research_Progress_Supervisor.pdf`.

**Scope.** Completed experiments through B2-E03 only (commit `cdd87b5`). ROOT is frozen at `d31864a`. The deck
contains no B2-E04 content, no unexecuted experiment presented as done, and no completion percentage.

| File | Role |
|---|---|
| `extract_deck_data.py` | Reads every number from committed CSVs/JSON **at `cdd87b5`** (`git show`) and writes `deck_data.json` |
| `build_deck.js` | pptxgenjs generator. All slide numbers and native charts come from `deck_data.json`; registers each displayed number in `displayed_numbers.json` |
| `check_consistency.py` | Read-only audit. Recomputes key values from the CSVs independently, checks method counts against the registry, and checks the scope/wording rules. Output: `consistency_check.txt` |
| `figures/` | Crops of two existing committed figures (`B2-E01_displacement_trace.png`, `B2-E02_q_of_t.png`). No new plots with typed data |

**Rebuild:**

```bash
python presentation/extract_deck_data.py                     # needs the repo venv (numpy, scipy)
NODE_PATH=<dir with pptxgenjs, react-icons, react, react-dom, sharp> node presentation/build_deck.js <pptx skill>/scripts/apply_theme.js
python presentation/check_consistency.py
```

**Before presenting, fill in the slide-1 placeholders** `[Presenter name]` and `[Programme / institution]`. The
repository metadata does not identify the presenter, so they were not guessed.
