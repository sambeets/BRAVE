# context.md - sipue-modelling (Codex working context)

## Current status (paper writing)
Paper drafting is active. The user manually updated `research/paper/paper.tex` and `research/paper/references/references.bib` and we will use these as the authoritative sources for writing.

### Model + results snapshot
- Canonical study: four archetypes x ten shared uncertainty states.
- Canonical cases: no_services, baseline, robust.
- Canonical results: `output/congested3/results_4_archetypes_10_states.csv`.
- KPI benchmark: `output/congested3/kpi_benchmark_summary.csv`.
- Validation outputs: `output/validation/train-test/` and `output/validation/out-of-sample/`.
- Creative figures: `output/congested3/figures/fig_*.pdf`.

### Writing workflow
- Use `research/paper/paper.tex` and `research/paper/references/references.bib` as the source of truth.
- Add new text and figures references directly to `paper.tex`.

---

## Notes / guardrails
- Do not use stale toy outputs or deleted result bundles as evidence.
- Keep Windows-friendly output (avoid unicode symbols if needed).
