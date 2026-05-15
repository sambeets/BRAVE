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
- **RQ3:** How do robust, profit-seeking design decisions shift traditional efficiency metrics and system-integrated metrics?

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

$$\text{siPUE} = \frac{[\text{E}_\text{FAC} - \text{E}_\text{credit}]^+}{\text{E}_\text{IT}}$$

where the energy-equivalent credit is:

$$E_\text{credit} = \frac{C_\text{heat}}{\bar{p}_\text{heat}} + \frac{C_\text{flex}}{\bar{p}_\text{flex}} - \frac{C_\text{CO2}}{\bar{p}_\text{CO2}}$$

- **When services are absent:** siPUE = PUE (backward compatible).
- **When services are active:** credits reduce effective energy burden.
- Credits are based on monetized, metered service revenues (conservative, auditable).
- The `[·]+` clip prevents negative effective energy use.

**Key Properties:**
- **Comparability:** Reduces to PUE when no services are credited.
- **Boundedness:** Energy-adjusted numerator clipped at zero.
- **Market-aligned auditability:** Credits scale with market prices and require settled revenues.

**Use Cases:**
- *Operators:* Dispatch evaluation for heat recovery, storage, flexibility.
- *Investment planning:* Design-stage signal alongside NPV/IRR for comparing efficiency upgrades vs. integration assets.
- *Lenders/investors:* KPI closer to cash-flow logic than PUE when services are active.
- *Policymakers:* PUE-compatible way to recognize system services in reporting.

---

### Two-Stage Optimization Framework

**Stage 1 — Scenario-Perfect Design:**  
For each scenario s, maximize profit selecting optimal design capacities and operations → yields upper-bound benchmark Π*_s.

**Stage 2 — Robust Shared Design:**  
Select a single design across all scenarios by minimizing a weighted combination of worst-case regret and average profit:

$$\min_{\hat{x}} \; \lambda \, Z(\hat{x}) - (1-\lambda)\bar{\Pi}(\hat{x})$$

with λ = 0.7 (risk-averse, prioritizes regret containment).

**Regret:** Gap between scenario-optimal profit and profit achieved by shared design.

---

### Optimization Model

**Objective (Eq. 3a):** Maximize profit = IT revenue + heat revenue + flexibility revenue − electricity cost − carbon cost − CAPEX − land cost − penalty cost.

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

### Overall Metric Alignment (Annual Rep-Days, 10 Scenarios)

- **PUE worsens in all 10 scenarios** under the robust integration design (Fail rate: 100%).
- **siPUE improves in all 10 scenarios** under the robust integration design (Pass rate: 100%).
- **Average profit uplift: +3.65%** (EUR ~3.01M average improvement).
- siPUE shows stronger rank association with unit profit than PUE (**ρ = 0.442 vs. 0.297**).

**Selected scenario results (baseline b → robust r):**

| Scenario | PUEb | PUEr | siPUEb | siPUEr | Profit% |
|---|---|---|---|---|---|
| Hyperscale-NL | 1.283 | 1.283 | 0.817 | 0.680 | +3.97% |
| AI-Norway | 1.358 | 1.359 | 0.878 | 0.753 | +5.27% |
| Streaming-Belgium | 1.331 | 1.332 | 0.885 | 0.724 | +5.44% |
| Hyperscale-Sweden | 1.305 | 1.305 | 0.841 | 0.710 | +3.05% |
| Streaming-UK | 1.375 | 1.376 | 0.975 | 0.756 | +0.37% |
| AI-Sweden | 1.349 | 1.350 | 0.920 | 0.814 | +2.19% |
| Colocation-Belgium | 1.330 | 1.330 | 0.902 | 0.814 | +3.37% |
| AI-Denmark | 1.360 | 1.361 | 0.858 | 0.728 | +7.65% |
| Streaming-France | 1.375 | 1.375 | 0.965 | 0.804 | +1.12% |
| Hyperscale-Germany | 1.285 | 1.286 | 0.815 | 0.652 | +4.03% |

### Sensitivity Analysis
- **CO₂ price ↑** → siPUE worsens (higher carbon exposure).
- **Heat price ↑** → siPUE improves (larger credits).
- **Flex price and congestion:** comparatively small effects under current calibration; heat and carbon dominate siPUE response.

### Construction Delay Sensitivity

| Δ | Avg Profit% | ΔPUE | ΔsiPUE | ΔsiPUE% | siPUE pass rate | PUE fail rate | KHX (MWth) |
|---|---|---|---|---|---|---|---|
| 0.00 | +0.40% | 0.001 | −0.137 | −15.31% | 1.00 | 1.00 | 23.91 |
| 0.25 | +0.37% | 0.002 | −0.104 | −10.04% | 0.90 | 0.90 | 22.31 |
| 0.50 | +1.29% | 0.002 | −0.066 | −5.67% | 0.80 | 0.70 | 19.50 |

As construction delay increases: optimal heat/flex capacities shrink, siPUE alignment rate drops, reinforcing that siPUE captures economic viability of services.

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
