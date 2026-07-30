# Context: Impact of Datacenter on Local Energy Systems

**Source:** Sambeet Mishra, Ankana Sinha, Khaled Aleikish, Thomas Øyvang (University of South-Eastern Norway), Dominic Stephen (Bax Innovation, Barcelona)  
**Funding:** BRAVE, FME RenewHydro (Grant 350468, Research Council of Norway)

---

## Overview

Data centers are rapidly becoming one of the most energy-intensive components of Europe's digital infrastructure. EU data center electricity consumption was ~76.8 TWh in 2018, reached ~96 TWh by 2024, and could approach 168 TWh by 2030 (up to ~5.7% of EU electricity by 2035). Beyond being large consumers, data centers can function as local energy-system resources when waste heat and controllable demand are integrated with surrounding networks.

The paper has two parts:
1. **Part I** – Policy, market, and project evidence on how energy characteristics and policy frameworks shape local business impacts across Europe.
2. **Part II** – Introduces **System-Integrated PUE (siPUE)**, a new metric and two-stage optimization framework for evaluating design and operational decisions under uncertainty.

---

## Research Questions

- **RQ1:** How do energy characteristics and policy frameworks shape local business impacts of data centers in Europe?
- **RQ2:** Can a unified metric capture the system value of heat and flexibility services while remaining comparable to PUE?
- **RQ3:** How do robust system-integrated planning decisions shift traditional efficiency metrics, siPUE, independent system benefit, and operator settlement outcomes?

---

## Part I: Data Centers in Local Energy Systems — Policy & Market Context

The investment landscape is structured across three governance levels:

### Macro Level — Policy and Market Architecture
- EU climate policy (European Green Deal, "Fit for 55") drives decarbonization, increasing value of efficiency, waste-heat reuse, and flexibility services.
- **Directive (EU) 2023/1791** (Energy Efficiency Directive recast) introduces formal reporting obligations (PUE, Renewable Energy Factor, Energy Reuse Factor) for large data centers.
- **Climate Neutral Data Centre Pact** sets sector-level renewable supply and heat-reuse targets.
- EU ETS carbon pricing creates regulatory risk and shapes the relative value of demand reduction and electrified heat pumps.
- Electricity market design determines whether flexibility can be monetized; open balancing markets allow data centers to participate via battery and load flexibility.

### Meso Level — Regional Infrastructure and Spatial Constraints
- Grid capacity is often the binding constraint; network operators have constrained/paused new connections in some regions due to local network limits.
- Congestion pricing and connection-rights frameworks make grid access a strategic asset.
- Land is a parallel constraint: integration equipment (heat pumps, exchangers, storage, substations) expands footprint and creates spatial lock-in once a site is committed.
- District heating infrastructure viability depends on network proximity, temperature requirements, and utility tariff structures.
- **Construction delay** is the critical meso-level risk: data centers build in 18–24 months while grid upgrades or district-heating expansions require multi-year cycles, stranding capital.

### Micro Level — Project Finance and Capital Allocation
- Integration adds CAPEX (heat pumps, exchangers, storage, telemetry) but diversifies revenue via heat offtake and flexibility payments.
- Uncertainty raises cost of capital; delay risk can breach debt-service covenants.
- Conventional **PUE** penalizes integration assets that use additional electricity to deliver external value, creating a **KPI gap** between performance metrics and actual system value.
- This gap leads to structural mispricing: private cost of capital does not reflect public value creation.

### Service Channels and Revenue Stacking

Three principal service channels:

1. **Heat Recovery**  
   - Secondary water loop extracts low-grade heat (25–40°C) from cooling circuits, upgraded by heat pump to district-heating supply levels.  
   - Commercial viability depends on tariff structures, network access, and long-term offtake contracts.  
   - Heat pricing often indexed to local fuel prices/carbon pricing.

2. **Electrical Flexibility**  
   - Controllable loads and batteries (originally for reliability) can provide balancing and ancillary services.  
   - Products: frequency regulation, reserve capacity, demand response/arbitrage.  
   - European data centers could offer substantial demand response if market design enables participation.

3. **Renewable Integration / Coordinated Anchoring**  
   - Corporate PPAs and siting near low-carbon generation.  
   - "Energy hub" concepts co-locate data centers with renewables and storage; data center as anchor demand for local balancing.

### Cross-Level Misalignment and the KPI Gap
- PUE classifies all non-IT electricity as overhead; integration assets can increase PUE even when creating external value.
- This misalignment is amplified by policy-driven reporting requirements (Directive 2023/1791).
- An effective KPI must be both **backward-compatible** (comparable to PUE) and **forward-compatible** (recognizing system-value creation).

---

## Part II: Methodological Contribution — siPUE and Optimization Framework

### System-Integrated PUE (siPUE)

**Definition:**

$$\text{siPUE}_s = \frac{[\text{E}_\text{FAC,s} - \text{E}_\text{sys,s}]^+}{\text{E}_\text{IT,s}}$$

where the system energy-equivalent credit is:

$$E_\text{sys,s} = a_s \frac{C^\text{ind}_\text{0,s} - C^\text{ind}_\text{s}}{p_\text{ref}}$$

- **When services are absent:** siPUE = PUE in the no-services counterfactual (backward compatible).
- **When services are active:** avoided independent system cost reduces effective energy burden; additional system cost increases it.
- \(a_s = (1-\Delta_\text{global})(1-\Delta_s)\) is the construction-availability factor, so delayed services receive only the system credit available during the operating fraction of the modeled horizon.
- \(p_\text{ref}\) is the configured EUR/MWh reference used to convert net system benefit into an MWh-equivalent credit.
- The independent system cost includes resource cost, transmission cost, carbon damage, peak/network cost, avoided heat-system cost, and avoided flexibility-system cost. Private heat/flex revenue is a transfer and does not directly enter the siPUE numerator.
- Cooling electricity is included in E_FAC, PUE, and siPUE. Heat recovered from IT equipment and cooling circuits can contribute to useful heat export, but cooling is not modeled as a separate external service or revenue stream.
- The `[·]+` clip prevents negative effective energy use.

**Key Properties:**
- **Comparability:** Reduces to PUE when no services are credited.
- **Boundedness:** Energy-adjusted numerator clipped at zero.
- **System-value auditability:** Credits are tied to an explicit no-services counterfactual and independent avoided system cost.

**Use Cases:**
- *Operators:* Dispatch evaluation for cooling-aware heat recovery, storage, flexibility, and carbon-market exposure, with settlement accounting retained only as a background output.
- *Investment planning:* Design-stage signal alongside NPV/IRR for comparing efficiency upgrades vs. integration assets.
- *Lenders/investors:* KPI closer to cash-flow logic than PUE when services are active.
- *Policymakers:* PUE-compatible way to recognize system services in reporting.

---

### Two-Stage Optimization Framework

**Stage 1 - Scenario-Perfect Design:**  
For each scenario s, solve the scenario-specific planning problem to obtain the benchmark planning-objective cost `C*_s`. Ex-post settlement fields remain available for audit/backward compatibility but are not used as manuscript diagnostics.

**Stage 2 - Robust Shared Design:**  
Select a single design across all scenarios by minimizing planning regret under the configured robust objective. Legacy code aliases may still use profit-oriented names, but manuscript-facing accounting should focus on planning objective cost, integrated planning surplus, and system-benefit metrics:

`PlanningRegret_s = C_s(shared design) - C*_s`

with `Z = max_s PlanningRegret_s`.

**Regret:** Gap between the scenario-perfect planning benchmark and the shared-design planning objective cost.

---
### Optimization Model

**Objective/accounting (Eq. 3a):** The optimization equations are kept unchanged in the current code, but manuscript-facing outputs are reported through `PlanningObjectiveCostEUR`, `IntegratedPlanningSurplusEUR`, and independent system-benefit fields. Ex-post settlement fields remain in the CSV for audit/backward compatibility, but they are not used as a diagnostic claim.

**Key Constraints:**
- Hourly energy balance (Eq. 3b): grid import + battery discharge = IT + overhead + heat export/COP + battery charge
- IT demand limit (Eq. 3c)
- Heat export bounded by heat exchanger capacity, DH export capacity, and recoverable heat fraction (Eq. 3d)
- Flexibility bounded by flexibility capacity (Eq. 3e)
- Battery state-of-charge dynamics (Eqs. 3f–3h)
- Land area as linear function of installed capacities (Eq. 3i)

**Network:**
- **Power:** 3-node DC-OPF (nodes: Germany N1, Netherlands N2, Great Britain N3); data center at N1.
- **Heat:** 2-node pipe (DC → DH node) with thermal storage, endogenous DH acceptance.

**Congestion:** Endogenous via DC-OPF line capacities; hourly utilization proxy for diagnostics.

**Land economics:** Linear land cost scaled to installed capacities and modeled horizon.

**Construction delays:** Revenue scaled by factor (1−Δ) where Δ is fraction of horizon with unavailable assets.

**Implementation:** Linear program using PuLP + HiGHS solver.

---

### Data and Scenario Design

- Hourly profiles from calibrated 3-node network time series (loads, renewable availability, fuel prices, optional CO₂ price).
- **7 representative days** selected with year-month method (preserving seasonal structure; T=8760 hours).
- **4 data-center archetypes x 10 shared uncertainty states = 40 services-active scenarios.** This replaced the earlier 10-named-country-scenario design (Hyperscale-Netherlands, AI-Norway, etc. -- that generator, `make_scenarios()`, still exists for legacy/test use but is no longer wired into the paper-facing canonical runner). The new design lets each archetype face the identical set of ten uncertainty draws.

| Archetype | Base IT (MW) | flex_frac | base_pue | heat_recovery_frac | hp_cop | land_limit_m2 |
|---|---|---|---|---|---|---|
| AI/HPC | 40 | 0.24 | 1.12 | 0.50 | 3.8 | 1500 |
| Hyperscale cloud | 60 | 0.16 | 1.18 | 0.45 | 3.5 | 1900 |
| Colocation | 25 | 0.10 | 1.30 | 0.32 | 3.1 | 1050 |
| Enterprise/edge | 8 | 0.05 | 1.42 | 0.20 | 2.8 | 620 |

**2026-07-23, later same day -- archetype table now cited (was previously bare literals).** Review of `heat_recovery_frac`, `hp_cop`, and `flex_frac` in `DATA_CENTER_ARCHETYPES` (`brave/scenario.py`) found that several archetype assumptions lacked direct sourcing, unlike the broader parameter set in `brave/provenance.py`. Literature search (ACEEE Nadel 2025 white paper, a 2015 multi-tenant colocation demand-response study, liquid-cooling market-adoption data) found: (a) `base_pue` and `hp_cop` archetype values were already within cited/generic ranges, no change; (b) AI/HPC's `heat_recovery_frac` exceeded the project's own already-cited generic range of 0.30-0.50 -- capped to 0.50; (c) Colocation's `flex_frac` was below the one directly relevant citation's floor (10-25% achievable, arXiv:1504.07308) -- raised to 0.10; (d) Enterprise/edge's `heat_recovery_frac` and `flex_frac`, and Hyperscale's `flex_frac`, have no archetype-specific source at all -- left unchanged and explicitly flagged as evidence gaps in `provenance.py` rather than assigned unsupported precise-looking numbers. Full citations and per-archetype rows are in `brave/provenance.py`; rationale is also in `brave/scenario.py`'s comment above `DATA_CENTER_ARCHETYPES`. Regenerate `main.py` and validation outputs before citing archetype-level numbers affected by these assumptions.

- **10 uncertainty states** (central-base, clean-cool, high-ets, heat-rich, flex-rich, cooling-stress, congestion-stress, storage-limited, cost-stress, service-upside) applied unchanged to every archetype, varying electricity/heat/flex/carbon prices, congestion, cooling overhead, storage availability, capex, and land cost via multipliers.
- The land representation is a stylized lease/opportunity-cost and feasibility diagnostic, not an engineering land-suitability model for integration-equipment sizing.
- Uncertainty: stochastic weekly perturbations to heat availability and flexibility prices.
- Sensitivity runners exist for diagnostic use, but their outputs should not be cited in the manuscript unless regenerated from the current source tree and reviewed separately.

---

## Key Results

**Current metric status:** the code defines siPUE from the no-services independent system-cost counterfactual:

`siPUE = [E_FAC - a_s (C_ind,0 - C_ind) / p_ref]^+ / E_IT`

Market heat/flex revenues and carbon-market payments remain dispatch and diagnostic outputs only; they are not direct siPUE numerator terms. Paper tables should use compact labels such as `CO2 cost`, `DC CO2 share`, `DC CO2 change`, `carbon damage`, `system benefit`, `benefit per MWh IT`, and `delay factor`.

For the manuscript, treat numerical results as illustrative until the canonical outputs are regenerated from the current source tree. Defensible claims are limited to the metric mechanics and property checks:

- siPUE equals PUE in the no-services case.
- A positive independent system-cost benefit lowers siPUE, subject to the configured nonnegative clip.
- Construction delay weakens the credited system benefit.
- PUE and siPUE can move in different directions because PUE counts integration electricity as overhead while siPUE credits independently valued system services.

Do not cite unsupported numerical or diagnostic claims from stale result files.

---
## Key Variables and Parameters

| Variable | Description |
|---|---|
| GRIDt | Grid electricity import at time t (MWh) |
| ITt | IT electricity served at time t (MWh) |
| HEATt | Heat exported to district heating at time t (MWhth) |
| FLEXt | Flexibility service provision at time t |
| BCHt, BDISt | Battery charge and discharge at time t (MWh) |
| SOCt | Battery state of charge at time t (MWh) |
| KHX | Heat exchanger capacity (MWth) |
| KFLEX | Flexibility capacity (MW) |
| KE_BAT, KP_BAT | Battery energy and power capacity |
| η | Efficiency improvement factor |
| p^IT_t | IT service revenue per delivered IT energy (EUR/MWh) |
| p^el_t | Electricity price at time t (EUR/MWh) |
| p^heat_t | Heat price at time t (EUR/MWhth) |
| p^flex_t | Flexibility price at time t (EUR/MW or EUR/MWh) |
| γt | Carbon factor at time t (kgCO₂/MWh) |
| p^CO2_t | Carbon price at time t (EUR/tCO₂) |
| ϕ | Recoverable heat fraction (dimensionless) |
| COPHP | Heat pump coefficient of performance |

---

## Limitations

1. Stylized scenarios and simplified representations; numerical results are not site-specific forecasts.
2. Finite scenario set with min–max regret (not full probabilistic/stochastic optimization).
3. Several real-world operational complexities are abstracted away.
4. siPUE is a performance indicator (not a physical efficiency ratio or replacement for detailed design optimization).

---

## Future Work

- Calibrate with real-world hourly electricity, ancillary-service prices, carbon factors, and observed IT load profiles.
- Couple optimization to district heating network constraints and temperature-dependent heat-pump performance.
- Replace congestion proxy with explicit network models (nodal pricing, distribution constraints).
- Extend to multi-actor coordination (contracting between operators, utilities, municipalities).
- Layer complementary indicators (water use, embodied carbon, location-based carbon accounting) alongside siPUE.

---

## Governance Summary Table

| Level | Typical functions |
|---|---|
| **Macro** (EU/national) | Climate targets, reporting requirements, carbon-price exposure, market-design rules for flexibility |
| **Meso** (regional/local) | Network planning, connection rights, congestion management, district-heating access, land/zoning/permitting, infrastructure sequencing |
| **Micro** (project) | Heat-offtake and flexibility contracts, metering/telemetry, operational control, risk allocation |
