# Hypotheses — green hydrogen production

This sweep revisits a set of green hydrogen hypotheses at the interface of materials, power systems, and infrastructure planning. The formulations below have been tightened to avoid over-claiming, to clarify where analogies rather than direct evidence are being used, and to flag key uncertainties in technology performance and demand trajectories.

All items remain hypotheses: they extrapolate from existing studies and data but are not established results. Each outlines a possible mechanism and suggests concrete ways to test or falsify it, and none should be taken as a basis for deployment decisions without further targeted evidence.

## H1 · Acid-mine drainage as an electrolyte and remediation target — score 0.89

*bridge · novelty 0.90 · plausibility 0.74 · support 0.92 · structure 1.00 · verified (confidence 0.86)*

**Hypothesis.** Deploying X20CoCrWMo10-9 steel–based OER anodes in PEM-like acidic electrolyzers at or near coal-mine tailings sites, operated directly in partially neutralized sulfuric-acid–contaminated drainage, could simultaneously produce green hydrogen and, under suitable conditions, reduce the ecotoxicity of acid mine drainage by converting some dissolved sulfate and metals into less bioavailable forms.

**Rationale.** Acid mine drainage from coal tailings often contains sulfuric acid and dissolved metals at levels that are toxic to aquatic ecosystems. Independently, cobalt-based stainless steel X20CoCrWMo10-9, after electro-oxidation, has been shown to act as a relatively stable, non‑noble oxygen-evolution electrocatalyst in dilute sulfuric acid, with acceptable weight loss over extended operation. Sulfuric acid is already a standard electrolyte for water electrolysis, and green hydrogen systems routinely operate in similar acidic environments. If acid mine drainage can be conditioned into a compatible acidity and impurity window, the same acidic medium might serve as both electrolyte and remediation target: oxygen evolution at X20CoCrWMo10-9 anodes could drive pH changes, co-precipitation, or oxidation of metal contaminants while renewable-powered electrolysis produces hydrogen. However, this remains speculative: electrochemical treatment might also generate more mobile or toxic species, change redox states in ways that increase bioavailability, or simply relocate contaminants into sludges that pose their own risks. No direct experimental evidence yet demonstrates a net reduction in ecotoxicity for this specific configuration, so the hypothesis is that there exists a parameter regime where net toxicity is reduced, not that such an outcome is guaranteed.

**Grounded in.**
- *sulfuric acid kills life in the rivers* — Environmental impact of electricity generation (https://en.wikipedia.org/wiki/Environmental_impact_of_electricity_generation) · corroborated by 1 source(s) · `relation:3837`
  > “Mill tailings are left out bare and have been leached into local rivers and resulted in most or all of the rivers in coal producing areas to run red year round with sulfuric acid that kills all life in the rivers.”
- *X20CoCrWMo10-9 steel used in sulfuric acid* — Electrooxidation of a cobalt based steel in LiOH: a non-noble metal based electro-catalyst suitable for durable water-splitting in an acidic milieu (http://arxiv.org/abs/1712.01100) · corroborated by 1 source(s) · `relation:14962`
  > “X20CoCrWMo10-9 steel, basi-cally consisting of Fe, Co, Cr, Mo and W, electro-oxidized in LiOH showed reasonable performance and stability towards long term usage as an OER electrocatalyst in 0.05 M H 2 SO 4 and was found to be highly competitive to recently developed and significantly more expensive OER electrocatalysts working in the acidic environment.”
- *X20CoCrWMo10-9 steel weight loss 98.9 µg mm -2 µg/mm2 (OER polarization 50 000 s, 0.05 M H2SO4, 10 mA/cm2)* — Electrooxidation of a cobalt based steel in LiOH: a non-noble metal based electro-catalyst suitable for durable water-splitting in an acidic milieu (http://arxiv.org/abs/1712.01100) · corroborated by 1 source(s) · `attribute:7604`
  > “Untreated steel X20CoCrWMo10-9 lost on average 98.9 µg mm -2 upon OER polarization for 50 000 s in 0.05 M H 2 SO 4 at 10 mA cm -2 (Table 1).”
- *sulfuric acid used as electrolyte* — Electrolysis of water (https://en.wikipedia.org/wiki/Electrolysis_of_water) · corroborated by 1 source(s) · `relation:55`
  > “Strong acids such as sulfuric acid (H2SO4), and strong bases such as potassium hydroxide (KOH), and sodium hydroxide (NaOH) are common choices as electrolytes due to their strong conducting abilities.”
- *sulfuric acid affects electrolysis efficiency* — Coalescence induced late departure of bubbles improves water electrolysis efficiency (http://arxiv.org/abs/2403.12064) · corroborated by 1 source(s) · `relation:12681`
  > “H2SO4 shows a marginal increase in efficiency with increasing concentration. This occurs because the addition of H2SO4 does not inhibit bubble coalescence but can improve solution conductivity.”
- *electrolysis produces green hydrogen* — From Policy to Practice: Upper Bound Cost Estimates of Europes Green Hydrogen Ambitions (http://arxiv.org/abs/2406.07149) · corroborated by 10 source(s) · `relation:2465`
  > “It includes requirements for the power supply to electrolysis: additionality (renewable energy from new installations within 36 months of the launch of the hydrogen plant), temporal correlation (production within the same hour as renewable powconsumption)n)and spatial correlation (location within the same power market bidding zone).”
- *steel X20CoCrWMo10-9 anode used in electrolysis of water* — Mid-infrared assisted water electrolysis as a method to substantially push down the overpotential of the oxygen evolution reaction (http://arxiv.org/abs/2301.05438) · corroborated by 1 source(s) · `relation:523`
  > “In order to show the effect of using Mid-IR radiation source on the water splitting capabilities of a steel anode (steel X20CoCrWMo10-9, see experimental section), water electrolysis was realized by the simplest approach imaginable from an experimental point of view upon using a glass beaker (single compartment conditions) and a three-electrode configuration consisting of a steel anode, a Pt counter electrode and a reversible hydrogen electrode.”

**How to test or refute.** 1) Characterize representative acid mine drainage streams (pH, sulfate, metals, organics) and bench-test X20CoCrWMo10-9 anodes in synthetic and real drainage adjusted toward 0.05 M H2SO4–like conditions, measuring OER activity, overpotential, corrosion rates, and catalyst poisoning over thousands of hours.
2) Track speciation of metals and sulfur during electrolysis (e.g., ICP-MS, XANES) to determine whether contaminants are immobilized (precipitation, adsorption) or transformed into less or more bioavailable forms, and assess changes in acute and chronic ecotoxicity using standard bioassays.
3) Build a pilot-scale electrolyzer loop at a mine site, operating on partially neutralized drainage with renewable power, and monitor hydrogen yield, electrode degradation, sludge composition, and downstream water quality over seasonal cycles.
4) Use techno-economic and life-cycle models to compare this integrated remediation–hydrogen concept with conventional treatment plus separate green hydrogen production, identifying parameter ranges (contaminant levels, pH, power prices) where the combined approach appears beneficial, neutral, or clearly counterproductive from an ecotoxicity standpoint.

## H2 · Oman as a liquid hydrogen aviation bunkering hub — score 0.84

*bridge · novelty 0.80 · plausibility 0.77 · support 0.76 · structure 1.00 · verified (confidence 0.78)*

**Hypothesis.** If Oman develops its planned green liquid hydrogen export hub with large-scale cryogenic shipping infrastructure, then by mid-century it could cost-competitively supply green liquid hydrogen to nearby Gulf aviation hubs for hydrogen-based aircraft, potentially positioning Oman as a specialized regional bunkering hub for aviation-grade liquid hydrogen rather than only a generic hydrogen exporter.

**Rationale.** Modeling suggests that Oman could achieve low-cost green hydrogen production with export-oriented infrastructure, including cryogenic liquid hydrogen chains, and separate analyses identify Oman as a promising future hub for green hydrogen production and export. Large-scale liquid hydrogen supply chains are already conceptualized around cryogenic storage and shipping. In parallel, aviation decarbonization studies for regions such as the Gulf and Europe explore scenarios in which hydrogen (in liquid form for long-haul) becomes an important aviation energy carrier, though these pathways and their timing remain highly uncertain and depend on aircraft technology, safety regulation, and competing options such as sustainable aviation fuels. The hypothesis extrapolates from these strands: if Oman’s export system is built around large-scale cryogenic liquid hydrogen chains compatible with aviation-grade specifications, then its geographic proximity and projected cost levels could make it an attractive supplier to Gulf airports. This would require that hydrogen-based aircraft reach commercial deployment at scale and that aviation demand for liquid hydrogen materializes in line with optimistic scenarios—assumptions that are not directly evidenced in the cited cost and supply-chain work and should be treated as contingent.

**Grounded in.**
- *Oman produces green liquid hydrogen* — Green Hydrogen Cost-Potentials for Global Trade (http://arxiv.org/abs/2303.00314) · corroborated by 1 source(s) · `relation:1367`
  > “The first time that costs for green hydrogen production drop below 2.50 EUR/kg occurs in Oman and Namibia in 2040.”
- *Oman considered as future hub for green hydrogen production and export* — Artificial Intelligence for Green Hydrogen Yield Prediction and Site Suitability using SHAP-Based Composite Index: Focus on Oman (http://arxiv.org/abs/2507.14219) · corroborated by 1 source(s) · `relation:8172`
  > “Oman is being considered as a future hub for green hydrogen production and export [1].”
- *large-scale liquid hydrogen supply chain stores cryogenic hydrogen* — Optimal supply chains and power sector benefits of green hydrogen (http://arxiv.org/abs/2005.03464) · corroborated by 1 source(s) · `relation:952`
  > “They mainly differ with respect to the form in which hydrogen is stored: gaseous hydrogen (GH 2 ), liquid hydrogen (LH 2 ), or bound to a liquid organic hydrogen carrier (LOHC, see [30]).”
- *cryogenic hydrogen instance of large-scale hydrogen supply* — Optimal supply chains and power sector benefits of green hydrogen (http://arxiv.org/abs/2005.03464) · corroborated by 1 source(s) · `relation:968`
  > “For each renewable energy share and hydrogen demand scenario, we combine the small-scale on-site hydrogen supply option with either of the three large-scale options.”

**How to test or refute.** 1) Develop integrated techno-economic models comparing delivered cost of liquid hydrogen from Oman to major Gulf airports versus alternative supply options (domestic production, other exporters), including liquefaction, storage boil-off, shipping, and last-mile logistics.
2) Combine these with aviation decarbonization scenarios from the literature (including hydrogen and non-hydrogen pathways) to estimate plausible ranges for timing, volume, and quality requirements for aviation-grade liquid hydrogen at Gulf hubs.
3) Assess compatibility between generic export-oriented cryogenic chains and aviation-specific requirements (purity, pressure, temperature, safety codes), identifying incremental infrastructure and cost needed for bunkering service.
4) Conduct sensitivity analyses on key uncertainties (electrolyzer and renewable capex, shipping costs, aircraft technology adoption, policy incentives, carbon prices) to map conditions under which Oman’s exports remain cost-competitive for aviation.
5) Engage in stakeholder and regulatory studies to evaluate whether airport–port integration (e.g., pipelines, trucking corridors) and safety frameworks can realistically be developed on the hypothesized timeline, and whether airlines and airports express credible demand for liquid hydrogen bunkering in the region.

## H3 · Hydropower–electrolyzer co-optimization to raise VRE penetration — score 0.83

*bridge · novelty 0.70 · plausibility 0.82 · support 0.77 · structure 1.00 · verified (confidence 0.86)*

**Hypothesis.** Coupling green hydrogen production via flexible electrolyzers to hydropower-dominated grids using a Pazouki-style stochastic optimization framework could increase the economically optimal penetration of variable renewables beyond what is achievable with hydropower balancing alone, by using hydrogen demand response to absorb residual variability in electricity demand and prices.

**Rationale.** In systems with high shares of wind and solar, hydropower is often the main dispatchable resource used to balance variability, but its flexibility is constrained by reservoir levels, inflows, and competing water uses. Pazouki-style models explicitly co-optimize flexible resources under uncertainty in wind, demand, and prices, treating them as controllable assets that respond to real-time conditions. If grid-connected electrolyzers are modeled as such flexible loads, their operation can be scheduled to consume electricity during low-price, surplus-renewable periods and reduce consumption during scarcity, effectively adding a new layer of demand-side flexibility. This additional controllable load should allow higher wind and solar shares before curtailment or reliability constraints become binding, compared with a system that relies solely on hydropower for balancing. The hypothesis is that, under realistic hydropower constraints, co-optimizing hydropower and electrolyzers shifts the cost-optimal capacity mix toward more variable renewables and hydrogen production, but the magnitude of this effect will depend on local hydrology, reservoir rules, and hydrogen value chains.

**Grounded in.**
- *hydro electricity supports electricity demand* — 100% renewable energy (https://en.wikipedia.org/wiki/100%25_renewable_energy) · corroborated by 1 source(s) · `relation:14161`
  > “hydroelectric can be called on to fill in the gaps.”
- *Pazouki et al. model models electricity demand* — Combination of Site-Wide and Real-Time Optimization for the Control of Systems of Electrolyzers (http://arxiv.org/abs/2404.06748) · corroborated by 1 source(s) · `relation:5668`
  > “The authors utilize a stochastic optimization approach, incorporating Monte-Carlo simulations to generate scenario trees based on predicted realtime pricing, wind, and electricity demand.”
- *Pazouki et al. model optimizes flexible energy resource* — Combination of Site-Wide and Real-Time Optimization for the Control of Systems of Electrolyzers (http://arxiv.org/abs/2404.06748) · corroborated by 1 source(s) · `relation:5666`
  > “Pazouki et al. [22] describe an optimization model for the operation of a system with energy generation from wind as well as the operation of flexible energy resources under uncertainty.”
- *hydropower (reservoir) supplies fuel/feedstock hydrogen* — Mapping Local Green Hydrogen Cost-Potentials by a Multidisciplinary Approach (http://arxiv.org/abs/2407.07573) · corroborated by 1 source(s) · `relation:1250`
  > “At lower levels of exploitation of the maximum potential, up to about 2%, the electricity required for the hydrogen production is supplied by hydropower (reservoir).”

**How to test or refute.** 1) Extend an existing hydropower-dominated system model by embedding a Pazouki-style stochastic optimization of flexible electrolyzers, calibrated with realistic technical constraints (ramp rates, minimum loads, efficiency curves) and hydrogen value chains.
2) Run scenario analyses comparing optimal capacity expansion and dispatch with and without flexible electrolyzers, measuring changes in cost-optimal wind/solar penetration, curtailment, system costs, and hydropower operation.
3) Perform sensitivity tests on reservoir constraints, inflow variability, hydrogen price, and electrolyzer flexibility to identify conditions under which the hypothesized increase in VRE penetration is most pronounced or disappears.
4) Validate model behavior against historical data from regions where hydropower and industrial electrolysis already coexist, if available, to check whether observed dispatch patterns align with model predictions.

## H4 · Stricter EU criteria favoring acid-based electrolysers — score 0.78

*bridge · novelty 0.80 · plausibility 0.80 · support 0.75 · structure 0.78 · verified (confidence 0.90)*

**Hypothesis.** If the EU’s green hydrogen criteria are tightened to enforce strong additionality and temporal and spatial correlation such that electrolysers must operate flexibly on variable renewable power, then in Europe the marginal value of proton-conducting, acid-tolerant electrolyser technologies could rise relative to alkaline systems, shifting the optimal technology mix toward acid-based electrolysers in high-renewables regions like Spain and Greece.

**Rationale.** AST90-style scenarios indicate that strict green hydrogen rules force electrolysers to run more flexibly on variable renewables and require higher electrolyser investment to maintain output compared with more relaxed cases. In the broader technical literature (beyond the specific sources listed here), proton-conducting, acid-tolerant technologies such as PEM cells are widely reported to be capable of higher current densities, faster ramping, and better dynamic response than conventional alkaline systems, which can be limited by mass transport and carbonate-related issues under frequent cycling. As policy constraints push grid-connected electrolysers into more variable, partial-load operation, the system-level cost of slower, less flexible alkaline units is likely to increase relative to more agile acid-based designs that exploit Brønsted–Lowry proton-transfer mechanisms. This effect should be strongest in nodes where strict criteria concentrate electrolyser capacity and renewable variability is high, such as Spain and Greece, potentially altering the optimal technology portfolio even if alkaline capex remains lower on a per-kW basis. Because the dynamic-performance comparison relies on general literature rather than the specific cited policy and chemistry sources, it should be treated as a qualitative expectation to be quantified in dedicated modeling.

**Grounded in.**
- *base applies to Europe* — From Policy to Practice: Upper Bound Cost Estimates of Europes Green Hydrogen Ambitions (http://arxiv.org/abs/2406.07149) · corroborated by 1 source(s) · `relation:2550`
  > “The first case, Base, envisions a Europe without requirements for the source of power supplied to electrolysis.”
- *AST90 increases investment in electrolyser* — From Policy to Practice: Upper Bound Cost Estimates of Europes Green Hydrogen Ambitions (http://arxiv.org/abs/2406.07149) · corroborated by 1 source(s) · `relation:2583`
  > “The second largest cost component is capital expenditure on electrolyzers. These require more flexible operation and higher capital investments to achieve the same production levels as constant output, as they cannot run on fossil power and depend on variable power sources.”
- *Brønsted–Lowry acid–base theory defines base* — Hydrogen (https://en.wikipedia.org/wiki/Hydrogen) · corroborated by 1 source(s) · `relation:11317`
  > “Under the Brønsted–Lowry acid–base theory, acids are proton donors, while bases are proton acceptors.”
- *AST90 concentrates electrolyzer capacity in Spain* — From Policy to Practice: Upper Bound Cost Estimates of Europes Green Hydrogen Ambitions (http://arxiv.org/abs/2406.07149) · corroborated by 1 source(s) · `relation:2601`
  > “In AST90, electrolyzer capacity is concentrated in fewer countries with competitive advantages in VRES production, such as Spain and Greece, due to their climate.”
- *AST90 concentrates electrolyzer capacity in Greece* — From Policy to Practice: Upper Bound Cost Estimates of Europes Green Hydrogen Ambitions (http://arxiv.org/abs/2406.07149) · corroborated by 1 source(s) · `relation:2602`
  > “In AST90, electrolyzer capacity is concentrated in fewer countries with competitive advantages in VRES production, such as Spain and Greece, due to their climate.”

**How to test or refute.** 1) Build or adapt a European power–hydrogen system model that distinguishes between alkaline and acid-based electrolysers with technology-specific capex, efficiency, ramping limits, and degradation under cycling, drawing dynamic parameters from the broader PEM vs alkaline literature.
2) Implement Base and AST90-like policy scenarios with varying degrees of additionality and temporal correlation, and optimize system costs and technology choices under each.
3) Compare the resulting electrolyser technology mix, regional siting (especially in Spain and Greece), and total system costs across scenarios to quantify how stricter criteria change the relative value of acid-based versus alkaline systems.
4) Conduct sensitivity analyses on technology learning curves, flexibility parameters, and policy design to identify thresholds where acid-based technologies become preferred.
5) Where possible, validate modeled flexibility advantages using operational data from existing PEM and alkaline plants operating under variable renewable supply.

## H5 · PENNs for site-specific VRE–electrolyzer degradation modeling — score 0.77

*bridge · novelty 0.80 · plausibility 0.84 · support 0.63 · structure 0.78 · verified (confidence 0.90)*

**Hypothesis.** Physics-enforced neural networks could be used to learn site- and technology-specific degradation and availability profiles of VRE-powered electrolyzers and embed them into additionality constraints analogous to Equation 1 in policy models, yielding more accurate and less uniformly conservative planning of VRE capacity per electrolyzer than static assumptions about availability and efficiency.

**Rationale.** Current policy-oriented planning equations, such as the Equation 1 used in the referenced EU green hydrogen study, link new VRE capacity to electrolyzer investments using constant specific power consumption and simplified availability assumptions, explicitly acknowledging that these simplifications are imperfect proxies for true additionality and may bias capacity requirements. In a separate membrane degradation study, a physics-enforced neural network is trained to output parameters of a physically motivated degradation profile, which are then plugged into that paper’s own Equation 1 to predict conductivity over time. The two Equation 1s are different formalisms in different domains; the connection here is an analogy in modeling structure, not a direct reuse. By analogy, a PENN trained on operational data from VRE-powered electrolyzers (including power variability, cycling, temperatures, and stack currents) could learn a small set of physically interpretable parameters describing effective availability, efficiency decay, and flexibility under local conditions. These parameters could replace or augment static coefficients in policy-planning equations, allowing additionality constraints to reflect site-specific behavior. The hypothesis is that such PENN-informed constraints would reduce over- or under-sizing of VRE relative to electrolyzer capacity, tightening planning around empirically grounded performance rather than generic capacity factors, but the degree of improvement remains to be demonstrated empirically.

**Grounded in.**
- *Equation 1 requires VRE* — From Policy to Practice: Upper Bound Cost Estimates of Europes Green Hydrogen Ambitions (http://arxiv.org/abs/2406.07149) · corroborated by 1 source(s) · `relation:2539`
  > “Equation 1 guarantees a minimum of new VRES investments per electrolyzer, acknowledging that VRES electricity is not available 100% of the time.”
- *Physics-Enforced Neural Network predicts Equation 1* — AI-Assisted Physics-Informed Predictions of Degradation Behavior of Polymeric Anion Exchange Membranes (http://arxiv.org/abs/2510.12655) · corroborated by 1 source(s) · `relation:6785`
  > “These parameters were then scaled to align the predicted values with the empirically observed ranges and finally plugged into Equation 1 along with each sample's recorded time value to get a conductivity prediction.”

**How to test or refute.** 1) Assemble datasets from multiple VRE-powered electrolyzer sites, including time series of input power, operating conditions, hydrogen output, and degradation indicators over multi-year periods.
2) Design a PENN architecture that encodes a physically motivated model of electrolyzer performance and degradation, with a small set of interpretable parameters constrained to realistic ranges, and train it on the collected data.
3) Compare PENN-derived effective availability and degradation parameters with those implied by current static assumptions, and evaluate predictive accuracy on held-out sites.
4) Integrate the PENN-derived parameters into a policy-oriented system model’s additionality equation (e.g., replacing constant specific power consumption with time- and site-dependent terms) and re-run capacity planning for representative regions.
5) Quantify differences in optimal VRE-to-electrolyzer ratios, system costs, and implied additionality across the PENN-based and static approaches, and test robustness via sensitivity analyses on data quality and model structure.

## H6 · Mainland–island co-optimization to expand islanded hydrogen — score 0.76

*bridge · novelty 0.70 · plausibility 0.84 · support 0.71 · structure 0.78 · verified (confidence 0.93)*

**Hypothesis.** Co-optimizing mainland electricity grid operation with a Pazouki-style stochastic control model for flexible electrolysers could increase the economically viable penetration of islanded green hydrogen production, by allowing electrolysers to provide short-term balancing services that reduce the need for mainland storage and peaking capacity.

**Rationale.** In current islanding analyses, the mainland grid typically serves exogenous electricity demand and may power integrated electrolysers, while islanded electrolysers rely on dedicated renewables and are treated more passively. Pazouki-style models, however, treat electrolysers as flexible resources whose operation is optimized under uncertainty in demand and wind generation. If both mainland-integrated and island-connected electrolysers (via hydrogen storage and reconversion) are co-optimized with the mainland grid, they can shift load in response to short-term fluctuations, absorbing surplus generation and backing off during scarcity. This added flexibility could reduce the need for batteries and peaking plants on the mainland, lowering system costs and making higher shares of islanded hydrogen production economically attractive. The hypothesis is that explicitly modeling this co-optimization will show a larger role for islanded hydrogen than models that treat electrolysers as inflexible or purely local loads, but the scale of the effect will depend on resource profiles, transmission limits, and hydrogen demand patterns.

**Grounded in.**
- *Pazouki et al. model optimizes flexible energy resource* — Combination of Site-Wide and Real-Time Optimization for the Control of Systems of Electrolyzers (http://arxiv.org/abs/2404.06748) · corroborated by 1 source(s) · `relation:5666`
  > “Pazouki et al. [22] describe an optimization model for the operation of a system with energy generation from wind as well as the operation of flexible energy resources under uncertainty.”
- *Pazouki et al. model models electricity demand* — Combination of Site-Wide and Real-Time Optimization for the Control of Systems of Electrolyzers (http://arxiv.org/abs/2404.06748) · corroborated by 1 source(s) · `relation:5668`
  > “The authors utilize a stochastic optimization approach, incorporating Monte-Carlo simulations to generate scenario trees based on predicted realtime pricing, wind, and electricity demand.”
- *mainland electricity grid serves electricity demand* — Benefits from Islanding Green Hydrogen Production (http://arxiv.org/abs/2310.12606) · corroborated by 1 source(s) · `relation:5790`
  > “The mainland electricity grid serves electricity demand and can also power integrated electrolysers.”

**How to test or refute.** 1) Extend an existing mainland–island power and hydrogen system model to include a Pazouki-style stochastic optimization of electrolyser operation on both mainland and island, with realistic technical constraints and hydrogen storage/reconversion pathways.
2) Compare capacity expansion and dispatch outcomes between a baseline with exogenous electrolyser operation and a co-optimized case, focusing on changes in mainland storage and peaking capacity needs and the share of hydrogen produced on islanded systems.
3) Perform sensitivity analyses on renewable resource profiles, transmission constraints between mainland and island, hydrogen demand levels, and electrolyser flexibility to identify conditions under which islanded production gains the most.
4) Where possible, benchmark model behavior against empirical data from systems that already use hydrogen or other flexible loads for balancing to validate the plausibility of the modeled balancing services.

## H7 · Modular operation of the Durazno solar electrolyzer plant — score 0.74

*bridge · novelty 0.60 · plausibility 0.85 · support 0.74 · structure 0.78 · verified (confidence 0.93)*

**Hypothesis.** Operating the 5 MW solar-powered electrolyzer plant in Durazno as a modular system could materially increase its ability to follow the variable output of the co-located 10 MW photovoltaic farm, enabling higher annual hydrogen yield per kW of electrolyzer capacity than a monolithic plant of the same nameplate size under the same solar and market conditions.

**Rationale.** The Durazno pilot couples a 5 MW electrolyzer to a 10 MW PV farm for green hydrogen production and heavy-truck refueling. Modeling work on renewable-electrolyzer plants suggests that modular configurations, where multiple smaller units can be independently dispatched, allow operators to approximate plant efficiency with a concave, piecewise-linear curve and effectively lower the minimum stable load. In such setups, some modules can be turned off while others operate near their efficiency sweet spot, improving part-load performance and enabling better tracking of variable renewable input. For a plant driven by an intermittent 10 MW PV profile, this modular flexibility should allow more of the solar energy to be converted into hydrogen rather than curtailed or used at inefficient low loads. The hypothesis is that, under realistic operating constraints, modular operation yields a measurable increase in annual hydrogen output per installed MW compared with a single-block electrolyzer of equal capacity, though the exact gain will depend on module sizing, control strategy, and local solar variability.

**Grounded in.**
- *electrolyzer plant located in Durazno* — Green hydrogen (https://en.wikipedia.org/wiki/Green_hydrogen) · corroborated by 1 source(s) · `relation:469`
  > “Hydrogen production will be carried out by a 5 MW electrolyzer plant located in Durazno, Uruguay, powered by a 10 MW solar farm.”
- *electrolyzer plant powered by photovoltaic solar farm* — Green hydrogen (https://en.wikipedia.org/wiki/Green_hydrogen) · corroborated by 1 source(s) · `relation:470`
  > “Hydrogen production will be carried out by a 5 MW electrolyzer plant located in Durazno, Uruguay, powered by a 10 MW solar farm.”
- *electrolyzer plant capacity 5 MW* — Green hydrogen (https://en.wikipedia.org/wiki/Green_hydrogen) · corroborated by 1 source(s) · `attribute:243`
  > “Hydrogen production will be carried out by a 5 MW electrolyzer plant located in Durazno, Uruguay, powered by a 10 MW solar farm.”
- *electrolyzer plant operates as modular system* — Electricity Market Bidding for Renewable Electrolyzer Plants: An Opportunity Cost Approach (http://arxiv.org/abs/2501.16844) · corroborated by 1 source(s) · `relation:13286`
  > “we assume that the electrolyzer plant is operated as a modular system, which enables us to neglect the lower load limit of individual electrolyzer modules [36].”

**How to test or refute.** 1) Develop detailed operational models of the Durazno plant under two configurations: a single 5 MW electrolyzer block and a modular arrangement (e.g., multiple smaller stacks) with realistic minimum load and efficiency curves for each module.
2) Simulate plant operation over a year using measured or high-resolution synthetic PV profiles and market conditions, optimizing dispatch in each configuration to maximize hydrogen yield or economic value.
3) Compare annual hydrogen production, curtailment, and utilization factors between the modular and monolithic cases, and perform sensitivity analyses on module size, minimum load, and efficiency characteristics.
4) If feasible, implement modular control strategies in the actual Durazno plant or a comparable pilot, and monitor real-world performance over time to validate model predictions.
5) Assess the incremental capex and O&M complexity of modularization versus the gains in hydrogen yield to determine net economic impact.

## H8 · Islanded PV/wind–electrolyser projects on U.S. public lands — score 0.72

*bridge · novelty 0.70 · plausibility 0.78 · support 0.60 · structure 0.78 · verified (confidence 0.78)*

**Hypothesis.** If the U.S. Interior Department prioritizes permitting islanded, off-grid renewable projects on suitable public lands or offshore areas that directly couple PV or wind to electrolysers without grid‑quality power electronics, then levelized hydrogen costs at those sites could be substantially lower than for grid‑connected, co-located electrolysers using the same resources, potentially enabling earlier cost-competitive green hydrogen production from Interior-managed projects.

**Rationale.** Large renewable projects already integrate electrolysers to absorb excess generation and produce green hydrogen, and power-converter-less PV–electrolyser architectures can dynamically match PV voltage to stack requirements, avoiding some DC–DC and DC–AC conversion stages and their associated costs and losses. Modeling of islanded electrolyser systems indicates that when grid-level power quality is not required, capital cost savings in power electronics can translate into sizable reductions in hydrogen cost compared with grid-integrated configurations, although the precise percentage reduction is site- and design-dependent and is not quantified in the specific evidence cited here. The U.S. Interior Department is mandated to permit large volumes of renewable capacity on public lands, many of which are remote or weakly connected to the grid, making direct-coupled, islanded configurations technically plausible. The hypothesis is that, by explicitly favoring such architectures in permitting and design, Interior-managed projects could achieve meaningfully lower levelized hydrogen costs than comparable grid-tied projects, but the magnitude of the cost advantage should be treated as an open empirical question rather than assumed a priori.

**Grounded in.**
- *renewable energy project integrates electrolyser* — Power System Architecture and Control for Green Hydrogen Production via Power Converter-less Photovoltaic-Electrolyser Integration (http://arxiv.org/abs/2407.10075) · corroborated by 1 source(s) · `relation:5426`
  > “Large renewable energy projects have integrated electrolysers, with the aim of extracting hydrogen to drive future power systems, such as in industries and hybrid transportation [5], [6], [7], [8].”
- *Interior Department of the United States committed to renewable energy project* — Renewable energy commercialization (https://en.wikipedia.org/wiki/Renewable_energy_commercialization) · corroborated by 1 source(s) · `relation:13816`
  > “He also mentioned the long-standing Interior Department commitment to permit 10,000 MW of renewable energy projects on public land in 2012.”

**How to test or refute.** 1) Identify representative Interior-managed sites with strong solar or wind resources and limited grid access, and design two project archetypes for each: (a) grid-connected renewables with co-located electrolysers and full power electronics, and (b) islanded, direct-coupled PV/wind–electrolyser systems with relaxed power-quality requirements.
2) Develop detailed techno-economic models for both archetypes, including capex for power electronics, efficiency losses, curtailment, and operational strategies, and compute levelized cost of hydrogen under consistent assumptions.
3) Perform sensitivity analyses on power electronics costs, electrolyser characteristics, resource variability, and potential revenue from grid services (for the grid-connected case) to map conditions under which islanded configurations are cheaper.
4) Evaluate regulatory, safety, and environmental constraints specific to public lands and offshore areas that might limit the feasibility of islanded configurations or require additional infrastructure.
5) If promising, implement one or more pilot projects under Interior’s permitting framework to empirically measure costs, reliability, and environmental impacts, and compare outcomes with model predictions.

Across these revised hypotheses, common themes are the search for integration synergies—using waste streams as electrolytes, aligning export chains with potential aviation demand, or co-optimizing electrolysers with grids and permitting regimes—and the recognition that many attractive narratives rest on untested assumptions. The adjustments here emphasize where evidence is indirect, where analogies are being drawn between different modeling frameworks, and where demand or technology pathways (notably in aviation and islanded systems) are especially uncertain. Systematic experiments, pilots, and transparent modeling will be essential to determine which of these ideas offer robust value versus context-specific or impractical concepts.


---
_8 hypotheses ranked · trace `traces/hypothesize_bridge.jsonl`_
