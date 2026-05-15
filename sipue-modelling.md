# context.md - sipue-modelling (Codex working context)

## Current status (paper writing)
Paper drafting is active. The user manually updated `research/paper/paper.tex` and `research/paper/references/references.bib` and we will use these as the authoritative sources for writing.

### Model + results snapshot (latest committed)
- 12 scenarios, country-only names; regimes: no_services / efficiency_only / services_active
- Regime mix: n_no_services=3, n_efficiency_only=3, services_active=6
- Annual run (365-day, profit_max) results: `toy_results_365day.csv`
- Short run (30-day, profit_max) results: `toy_results_30day.csv`
- Scenarios: `toy_scenarios.csv`
- Figures (PDF): `sipue-modelling/paper-figures/*.pdf`

### Writing workflow
- Use `research/paper/paper.tex` and `research/paper/references/references.bib` as the source of truth.
- Add new text and figures references directly to `paper.tex`.

---

## Notes / guardrails
- Preserve 12 scenarios and 3 regimes.
- Keep Windows-friendly output (avoid unicode symbols if needed).
