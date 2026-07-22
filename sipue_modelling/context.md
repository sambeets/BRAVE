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
- PUE classifies all non-IT electricity as overhead; integration assets worsen PUE even when creating external value.
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
- **10 services-active scenarios** (stylized European contexts):

| Scenario | Description |
|---|---|
| Hyperscale-Netherlands | Hyperscale site (Netherlands) |
| AI-Norway | AI-oriented site (Norway) |
| Streaming-Belgium | Streaming-oriented load (Belgium) |
| Hyperscale-Sweden | Hyperscale site (Sweden) |
| Streaming-United Kingdom | Streaming-oriented load (United Kingdom) |
| AI-Sweden | AI-oriented site (Sweden) |
| Colocation-Belgium | Colocation site (Belgium) |
| AI-Denmark | AI-oriented site (Denmark) |
| Streaming-France | Streaming-oriented load (France) |
| Hyperscale-Germany | Hyperscale site (Germany) |

- Uncertainty: stochastic weekly perturbations to heat availability and flexibility prices.
- OAT sensitivity: price multipliers (0.5×/2×) for CO₂, heat, flex; congestion multipliers (1.5×/0.5×); construction-delay factors Δ ∈ {0, 0.25, 0.5}.

---

## Key Results

**Current metric status:** the code defines siPUE from the no-services independent system-cost counterfactual:

`siPUE = [E_FAC - a_s (C_ind,0 - C_ind) / p_ref]^+ / E_IT`

Market heat/flex revenues and carbon-market payments remain dispatch and diagnostic outputs only; they are not direct siPUE numerator terms. Paper tables should use compact labels such as `CO2 cost`, `DC CO2 share`, `DC CO2 change`, `carbon damage`, `system benefit`, `benefit per MWh IT`, and `delay factor`.

### Overall Metric Alignment (Annual Rep-Days, 10 Scenarios)

Current canonical outputs are in `output/results_365day.csv` and `output/validation/validation_summary.csv`. The validation now separates three claims:

- **Headline percentage changes:** PUE worsens by `+0.309%` on average, while siPUE improves by `-24.961%` on average. Integrated planning surplus changes by `+0.085%`, and independent system benefit equals `0.255%` of the no-services independent system cost on average.
- **Planning-objective alignment:** siPUE is directionally aligned with planning improvement in all scenarios, but it should not be used to rank integrated planning surplus magnitudes in the current canonical run (`rho = -0.903` for siPUE improvement versus unit integrated-planning-surplus improvement).
- **External system validity:** siPUE strongly ranks normalized independent system benefit in the current canonical run (`rho = +0.988` against `SystemBenefitPerMWhIT_no_services_minus_rob`). Absolute independent system benefit remains a separate scale/impact metric and should not be treated as the primary siPUE rank-validity target.

Manuscript claims should therefore say that siPUE is a no-services-counterfactual system-integrated KPI that tracks system-adjusted burden per unit IT energy.

### Sensitivity Analysis

Under the updated formulation, price sensitivities should be interpreted through dispatch, integrated planning surplus, and independent system cost. A heat or flexibility price change affects siPUE only if it changes the independent no-services-vs-case system-cost counterfactual.
- **CO2 price increase:** changes carbon-market cost and carbon-damage diagnostics; in the current canonical run CUE increases by `+0.606%` versus no-services.
- **Heat price increase:** changes operator settlement economics and may change dispatch; siPUE changes only through the independent system-cost counterfactual.
- **Flex price and congestion:** should be interpreted through independent system benefit and planning-surplus diagnostics separately.
### Construction Delay Sensitivity

Construction-delay sensitivity should be regenerated with `run_sensitivity_prices_congestion_repdays_annual.py`. The output table now reports siPUE change, independent system benefit, integrated planning surplus, PUE, ERE, and CUE. Do not reuse older profit-centered or pass-rate tables in the manuscript.

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
