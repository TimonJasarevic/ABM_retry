# Finished ABM report and analysis

Open `report.pdf` for the compiled report (8 main-text pages, followed by ODD+ and supplementary results). `report.tex` is the entry point; `foundation.tex`, `supplement.tex` and `additional_results.tex` are included sources. `original_draft.tex` preserves the supplied draft. Literature citations and bibliography were omitted at the author's request; author names were not supplied and are not fabricated.

## Reproduce from saved data — no simulations

From `heterogeneous_implementation/`:

```bash
.venv/bin/python experiments/report_analysis.py --reuse
.venv/bin/python experiments/audit_sensitivity.py
.venv/bin/python experiments/report_tables.py
cd report
pdflatex -interaction=nonstopmode -halt-on-error report.tex
pdflatex -interaction=nonstopmode -halt-on-error report.tex
```

The sensitivity audit reads saved data and recomputes aggregation/Sobol estimators only. **It does not run any OFAT or Sobol model evaluations.** No sensitivity simulations were run during this work.

The separate `report_analysis.py` script executed exactly one new baseline trajectory: random regular network, N=120, degree=4, seed=42, 50,000 cascades, depth=20, burn-in=0, and explicit `rewire_prob=0.001`. Current behavioral defaults were retained. The script refuses to overwrite an existing history unless used in read-only simulation-reuse mode. To intentionally run another trajectory, first select a new output path/settings in the script; that is future work, not part of this delivery.

## Deliverables

- `report.tex`, three included `.tex` files, and compiled `report.pdf`.
- `../experiments/report_analysis.py`: baseline capture, incoming trust, correlations, dispersion, temporal blocks, figures.
- `../experiments/audit_sensitivity.py`: checks complete saved designs, hashes, seed aggregation, OFAT ranking, Sobol estimates/uncertainties; produces sensitivity tables and figures.
- `../experiments/report_tables.py`: generates supplementary LaTeX tables from saved CSVs.
- `../results/baseline_exploratory_seed42/history.csv`: all 50,000 cascades and original simulation metrics.
- `initial_nodes.csv`, `final_nodes.csv`, initial/final edge lists, `final_current_dyads.csv` in that same directory.
- `correlations.csv`, `heterogeneity.csv`, `trust_matched_connected.csv`, `action_blocks.csv`, `influencer_comparison.csv`, `simulation_summary.csv`, `summary.json`, and `manifest.json` there.
- Four new baseline figures, each PNG and PDF: dynamics, structure, heterogeneity, node_associations.
- Three sensitivity figures from existing data, each PNG and PDF: ofat_curves, sobol_burden, sobol_topology.
- `tables/`: full sensitivity indices, ranges, OFAT rankings and curves. `sensitivity_audit.json` and `validation.json` record checks.

No behavioral source files or existing sensitivity output files were changed. Existing unrelated deleted/untracked files in the workspace were left alone.

## Existing experiments used

1. `results/ofat_screening_rewire0`: 21 parameters × 3 levels × 3 seeds = 189 stored evaluations; 10,000 cascades, seeds 42–44; rewiring off.
2. `results/ofat_screening_rewire001`: same structure, rewiring 0.001.
3. `results/sobol_no_rewire`: 13 inputs, base size 64, 960 evaluations, 5,000 cascades, seed 42, rewiring off.
4. `results/sobol_adaptive_14_n64_rewire001`: **14**, not four, inputs; base size 64, 1,024 evaluations, 5,000 cascades, seed 42, rewiring 0.001.
5. `results/sobol_topology_gini_8_n32_8k`: 8 inputs, base size 32, 320 evaluations, 8,000 cascades, seed 42; output is mean degree Gini over cascades 6,401–8,000. Rewiring probability varies.

The author confirmed intentional comparability of the static/adaptive burden screens after omitting an unimportant parameter to reduce cost. Shared input ranges agree. The adaptive design fixes receiver_true_reward and adds rewire_sensitivity/global_search_prob. Comparisons of broad mechanism importance are valid as intended; precise indices condition on different complete input spaces. No claim that a small between-design index difference is caused by rewiring is made.

## Main findings

- OFAT sharing-reward/false-sharing-cost ranges: 0.551625/0.511284 static and 0.405013/0.373239 adaptive. Higher sharing reward raises burden; higher false-sharing cost lowers it. Verification cost, loss aversion and risk aversion are secondary.
- Sobol adaptive S1/ST: false-sharing cost 0.298376/0.891933; sharing reward 0.272303/0.477137; audience sensitivity 0.063080/0.147725. Static counterparts: 0.478167/0.680884, 0.194715/0.432023, and −0.002713/0.154582. Larger total-order estimates suggest interactions, with wide intervals; no pairwise effects are identified.
- OFAT and Sobol agree on payoff dominance. Audience sensitivity is weak in OFAT but has a larger, uncertain global total-order estimate. Local ranges and global variance shares are different quantities.
- Topology pilot S1/ST: rewiring rate 0.248782/0.606997, global search 0.167073/0.507487, rewiring sensitivity 0.153436/0.309767. It is preliminary, output-specific evidence; these indices are not misinformation sensitivity.
- Baseline actions stay near SHARE 45.6%, VERIFY 40.3%, DISCARD 14.1%, without a clear qualitative regime shift. This is descriptive stability, not a formal stationarity claim.
- Final degree SD 3.991658, CV 0.997914, Gini 0.539410, maximum 15, maximum/(N−1) 0.126050, minimum 0. There are 28 isolates, 18 degree-one nodes, 30 components, and 90/120 agents in the largest component; 4,796 successful rewires preserve 240 edges.
- Connectivity is unequal and fragmented, not winner-take-all. Freeman degree centralization is 0.094004. Relative hubs emerge, but calling the whole network only moderately differentiated would understate isolation.
- Spearman correlations: degree–reputation +0.217931 (n=120); degree–incoming trust +0.589832 (n=92); degree–deception −0.166917 (n=120); reputation–deception −0.224516 (n=120); incoming trust–deception −0.111894 (n=92). Incoming-trust analyses exclude exactly 28 isolates; others retain them.
- Reputation SD rises 0.109946→0.205755, deception SD 0.102088→0.208766. Outgoing effective-trust population SD rises 0.110002→0.159347 **with isolate defaults included**; among the same 92 ultimately connected agents it falls 0.111348→0.102834. Unqualified claims that all trust heterogeneity increased were rejected.

## Corrections to the original report

- Replaced the claimed primary conditional fake propagation outcome with the actual **unconditional fake exposure burden** used in saved sensitivity CSVs. Both metrics are now defined distinctly.
- Removed duplicate sensitivity headings/paragraphs, image placeholders, unfinished discussion/availability boilerplate and unsupported generic conclusions.
- Kept sender/originator payoff accounting distinct from audience-response production adaptation. It is not a payoff-learning model.
- Corrected the adaptive Sobol directory/input count, described the intentionally different designs, and added the separate topology pilot.
- Added one-update-per-cascade reputation timing, Gamma aversion draws, trust retention after edge removal, isolate behavior, candidate selection before dropping an edge, influencer tie handling and precise output definitions to ODD+.
- Replaced qualitative expectations with measured baseline results, especially extensive isolation and the trust-dispersion qualification.
- Replaced unconditional interaction and rewiring claims with range-, output- and uncertainty-qualified interpretations.

## Remaining limitations and provenance

All topology source hashes match current files. Adaptive burden's behavioral network/agent/definition hashes match; simulation and plotting hashes differ. Static burden has an older network hash; the exact old source was unavailable. OFAT has no saved manifest/hashes, although CSVs, PNG labels and the current script are internally consistent. Full historical execution equivalence cannot be certified. The stale pre-existing results README describes missing older experiments and was not used as evidence.

The baseline is one seed, with dependent nodes and co-evolving final-state variables. It supports no causal or robustness claims. Sobol bootstrap intervals are conditional on one model seed, and base designs are small. Parameter ranges, different horizons, low rewiring rate, conserved edges, isolate recovery rules, verification-only trust learning, global reputation, binary truth and independent cascades constrain interpretation. No empirical calibration or validation is claimed. Future simulations proposed in the report were **not** performed.
