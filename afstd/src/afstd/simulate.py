"""Synthetic placebo-controlled trials with known truth, in the AF-STD input schema (for tests and tutorials)."""
import numpy as np
import pandas as pd


def synthetic(n_drugs=30, n_events=25, n_trials=240, seed=1, tau=0.2, null_share=0.4, open_label_effect=0.3):
    rng = np.random.default_rng(seed)
    socs = [f"soc{i}" for i in range(5)]
    events = pd.DataFrame(dict(event=[f"event {i}" for i in range(n_events)], soc=[socs[i % 5] for i in range(n_events)]))
    drugs = pd.DataFrame(dict(drug_id=[f"D{i}" for i in range(n_drugs)],
                              atc=[[f"A0{i % 4}B{i % 7}"] for i in range(n_drugs)], targets=[[f"T{i % 6}|INHIBITOR"] for i in range(n_drugs)]))
    base = rng.normal(-3.2, 0.8, n_events)                        # event background (cloglog, 12 weeks)
    bt = rng.normal(0, .3, n_events); bd = rng.normal(0, .2, n_drugs)
    psi = np.where(rng.random((n_drugs, n_events)) < null_share, 0.0, 0.4 + bt[None, :] + bd[:, None] + rng.normal(0, .5, (n_drugs, n_events)))
    arms, trials = [], []
    for j in range(n_trials):
        d = rng.integers(n_drugs); n = int(rng.integers(40, 300)); open_ = rng.random() < .1
        trials.append(dict(trial_id=f"T{j:04d}", start_date=f"{2005 + j % 15}-06-01", followup_weeks=float(rng.choice([6, 12, 26, 52])),
                           masking="NONE" if open_ else "DOUBLE", sponsor_class="INDUSTRY", enrollment=2 * n))
        evs = rng.choice(n_events, size=int(rng.integers(5, 15)), replace=False)
        for e in evs:
            a0 = base[e] + rng.normal(0, .4)
            z = psi[d, e] + open_label_effect * open_ + rng.normal(0, tau)
            x0 = rng.binomial(n, -np.expm1(-np.exp(a0))); x1 = rng.binomial(n, -np.expm1(-np.exp(min(a0 + z, 3))))
            arms.append(dict(trial_id=f"T{j:04d}", arm_id="P", role="placebo", drug_id=None, event=f"event {e}", affected=x0, at_risk=n))
            arms.append(dict(trial_id=f"T{j:04d}", arm_id="A", role="drug", drug_id=f"D{d}", event=f"event {e}", affected=x1, at_risk=n))
    truth = pd.DataFrame([(f"D{i}", f"event {e}", psi[i, e]) for i in range(n_drugs) for e in range(n_events)],
                         columns=["drug", "event", "psi_true"])
    return pd.DataFrame(arms), pd.DataFrame(trials), drugs, events, truth
