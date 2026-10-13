# afstd — reference implementation of AF-STD

AF-STD specifies how to estimate, from placebo-controlled randomized trials, how much of a reported adverse event a drug
actually causes (the randomized excess or attributable fraction), how to validate an estimator, and how to report it.
`afstd` is the reference implementation of the estimator (AF 3.x); its model core is vendored verbatim from the benchmarked
implementation (`tools/vendor_core.py` records the source hashes, and `tests/` checks the two are identical).

## Quick start

```python
import afstd
rec = afstd.Records.from_frames(arms, trials, drugs, events)      # plain pandas tables, schema below
model = afstd.fit(rec, version="3.2")                             # empirical Bayes on a log-HR grid
table = afstd.estimates(model)                                    # one row per drug x event
Kt, q = afstd.predict_trial(model, afstd.Records.from_frames(new_arms, new_trials, drugs, events))
```

`estimates()` returns, per drug x event: the reference-trial log hazard ratio (`psi_*`), the hazard-scale excess fraction
(`af_hz`, 95% credible interval), the 12-week risk-scale excess fraction (`af_risk12`), the probability that the event is
mostly background (`p_af_lt_half`), a 95% predictive interval for a new population (`pred_lo`, `pred_hi`), and the number
of trials with direct evidence (`n_trials`; 0 = predicted from the structured prior only).

## Input schema

| table | required columns | optional |
|---|---|---|
| arms | trial_id, arm_id, role ('placebo'/'drug'), event, affected, at_risk; drug_id for drug arms | dose, dose_unit |
| trials | trial_id | start_date, followup_weeks, masking, sponsor_class, phase1_only, healthy_volunteers, mean_age, pct_female, enrollment |
| drugs | drug_id | atc (list), targets (list of 'target|action') |
| events | event, soc | |
| assessment | trial_id, event, nonsystematic_share | (whole table optional) |

An event contributes from a trial only if the trial reports it for every placebo and drug arm (registry convention).
Post-randomization variables (dropout, adherence) are deliberately not accepted as covariates.

## Validation

The estimator is validated with BENCH-2.1 (rolling-origin prediction of new trials' drug-arm counts, proper scoring,
calibration, strong baselines including Bayesian random-effects meta-analysis) and by known-truth simulation; see
`docs/AF-STD_spec` in the project for the protocol and the current results. Run the package tests with
`python -m pytest tests/`.
