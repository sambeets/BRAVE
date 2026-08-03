# sipue_modelling

Modeling code and experiment outputs for the paper:
**"Evaluating Data Centers as Local Energy System Assets: A System-Integrated PUE Approach"**
by Sambeet Mishra, Ankana Singha, Khaled Aleikish, Thomas Øyvang, Dominic Stephen, and Rob Hovsapian.

## Overview

This repository implements a two-stage optimization framework for data center integration with local energy systems and evaluates **System-Integrated PUE (siPUE)** alongside conventional PUE, integrated-planning accounting, and operator settlement diagnostics.

The core idea is to evaluate data centers not only as electricity loads, but as active system participants through:
- heat export to district heating,
- flexibility services,
- storage-enabled operation,
- and carbon, peak, network, heat, and flexibility effects in dispatch, integrated planning surplus, independent system cost, and validation outcomes.

## What This Repo Contains

- `brave/`: core modeling package (config, scenario generation, optimization, metrics, plotting, exports).
- `main.py`: thin compatibility entrypoint that re-exports public APIs from `brave`.
- `run_manuscript_sensitivity_repdays_annual.py`: optional manuscript sensitivity runner.
- `run_carbon_cap_relaxed_ablation.py`: optional carbon-cap diagnostic used to separate cap compliance from priced carbon-damage valuation.
- `data/network_3node_timeseries.csv`: network and time-series input data.
- `output/`: generated results tables and paper figures.
- `tests/`: unit and integration-style tests plus golden references.

## Model outline

From the referenced paper's abstract and setup:
- siPUE is defined to recover conventional PUE in the no-services counterfactual.
- siPUE uses delay-adjusted independent avoided system cost, not private heat or flexibility revenue:
  `siPUE = [E_FAC - a_s * (C_ind,0 - C_ind) / p_ref]^+ / E_IT`.
- The availability factor is `a_s = (1 - delta_global)(1 - delta_s)`, so construction delay reduces credited system benefit.
- Heat, flexibility, carbon damage, peak capacity, transmission, and resource-cost effects enter siPUE through the independent system-cost counterfactual.
- Private heat/flex revenues and carbon-market payments remain dispatch/profit and diagnostic outputs, but they do not directly enter the siPUE numerator.
- Cooling electricity is included in facility load, PUE, and siPUE. The recoverable heat source includes IT load and cooling heat, but the current formulation does not model a separate cooling export service or cooling-service revenue.
- Carbon market exposure is included in dispatch and profit through scenario or time-series CO2 prices.
- Carbon outcomes are also reported through emissions, CUE, carbon market cost, independent carbon damage, and independent system-cost outcomes so siPUE can be checked against physical and system-level diagnostics.
- Validation tables currently report metric property checks and selected diagnostics. Broader validation claims should be regenerated and reviewed before manuscript use.
- A normalized pure min-max regret formulation is reported as the integrated planning objective. siPUE-regret and cap-relaxed runs remain separate diagnostics.
- In services-active scenarios, siPUE should be interpreted as a system-integrated metric based on the no-services independent system-cost counterfactual. Absolute system benefit remains a scale/impact diagnostic.
- Paper-facing tables should use compact labels such as `CO2 cost`, `DC CO2 share`, `DC CO2 change`, `carbon damage`, `system benefit`, `benefit per MWh IT`, and `delay factor`.

## Setup and Run

### 1. Install dependencies

```bash
poetry install
```

### 2. Run base workflow

```bash
poetry run python main.py
```

The base workflow exports the paper-facing core cases: no-services, baseline, and robust normalized pure min-max design. Heavier sensitivity, out-of-sample, carbon-cap, and service-ablation diagnostics are kept as separate optional scripts.

### 3. Optional diagnostic sensitivity analysis

```bash
poetry run python run_manuscript_sensitivity_repdays_annual.py
```

This runner reports the manuscript sensitivity set with both labels and
numeric settings: heat price multiplier `0.30/0.90`, flexibility price
multiplier `0.125/0.375`, cooling overhead `0.8x/1.2x`, land availability
`0.65x/1.2x/100x`, and construction delay `0/12` months. Carbon-damage
valuation is kept as a separate policy ledger diagnostic, not a main
physical/market sensitivity factor.

### 4. Optional carbon-cap relaxed diagnostic

```bash
poetry run python run_carbon_cap_relaxed_ablation.py
```

This re-solves the baseline and two-stage robust design with the physical
carbon cap removed while keeping the carbon-damage price term active. The
output `output/congested3/carbon_cap_relaxed_ablation.csv` is used to check
whether the carbon-damage credit is driven by priced valuation or by cap
compliance.

### 5. Optional out-of-sample diagnostic

This designs the robust portfolio on scenarios 1-5 and evaluates the fixed
design under held-out scenarios 6-10.

```bash
poetry run python run_out_of_sample_validation.py
```

### 6. Optional train/test scenario diagnostic

This designs the robust portfolio on uncertainty states 1-7 and evaluates the
fixed design on held-out states 8-10, per archetype.

```bash
poetry run python run_train_test_scenario_validation.py
```

### 7. Run tests

```bash
poetry run pytest
```

## Workflow

```mermaid
flowchart TD
    A[Define planning objective and scenario set<br/>siPUE / PUE / operator diagnostics]
    B[Prepare inputs and assumptions<br/>data/network_3node_timeseries.csv + config]
    C[Run baseline optimization<br/>poetry run python main.py]
    D[Generate baseline outputs<br/>results + paper figures in output/]
    E[Optional diagnostics<br/>sensitivity and validation scripts]
    F[Review diagnostic outputs<br/>before manuscript use]
    G[Interpret and compare findings<br/>only from source-consistent outputs]

    A --> B --> C --> D
    D --> E --> F --> G
```

## Notes

- Python dependency management is configured via Poetry (`pyproject.toml`).
- Generated analysis artifacts are stored under `output/`.
