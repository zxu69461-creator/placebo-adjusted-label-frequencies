import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import afstd  # noqa: E402
from afstd import core  # noqa: E402
from afstd.simulate import synthetic  # noqa: E402


@pytest.fixture(scope="module")
def data():
    return synthetic()


def test_schema_rejects_bad_input(data):
    arms, trials, drugs, events, _ = data
    bad = arms.copy(); bad.loc[0, "affected"] = bad.loc[0, "at_risk"] + 1
    with pytest.raises(ValueError):
        afstd.Records.from_frames(bad, trials, drugs, events)
    with pytest.raises(ValueError):
        afstd.Records.from_frames(arms.drop(columns="role"), trials, drugs, events)


def test_records_shapes(data):
    arms, trials, drugs, events, _ = data
    rec = afstd.Records.from_frames(arms, trials, drugs, events)
    assert len(rec.K) == (arms.role == "drug").sum()
    assert rec.K.open_label.isin([0.0, 1.0]).all()
    assert set(rec.cf.columns) >= set(core.FEATS)


@pytest.mark.parametrize("version", ["3.0", "3.2", "3.3"])
def test_known_truth_recovery(data, version):
    arms, trials, drugs, events, truth = data
    rec = afstd.Records.from_frames(arms, trials, drugs, events)
    model = afstd.fit(rec, version=version, outer=2, inner=4)
    est = afstd.estimates(model).merge(truth, on=["drug", "event"])
    rich = est[est.n_trials >= 3]
    cov = ((rich.psi_true >= rich.psi_lo) & (rich.psi_true <= rich.psi_hi)).mean()
    assert 0.85 <= cov <= 1.0, cov
    rmse = np.sqrt(((rich.psi_mean - rich.psi_true) ** 2).mean())
    assert rmse < 0.4, rmse
    # the open-label bias is recovered with the right sign
    j = core.COVS.index("open_label")
    assert model["fit"]["beta_g"][j] > 0.1


def test_predict_is_a_density_and_handles_unseen(data):
    arms, trials, drugs, events, _ = data
    rec = afstd.Records.from_frames(arms, trials, drugs, events)
    model = afstd.fit(rec, version="3.2", outer=1, inner=3)
    new = arms[arms.trial_id == arms.trial_id.iloc[0]].copy()
    new["trial_id"] = "NEW"; new.loc[new.role == "drug", "drug_id"] = "D_new"
    rec_new = afstd.Records.from_frames(new, pd.DataFrame(dict(trial_id=["NEW"])),
                                        pd.concat([drugs, pd.DataFrame(dict(drug_id=["D_new"], atc=[["A01B1"]], targets=[["T1|INHIBITOR"]]))]),
                                        events)
    Kt, q = afstd.predict_trial(model, rec_new)
    assert np.allclose(q.sum(1), 1.0)
    assert (~Kt.seen).all()


def test_core_matches_project_implementation():
    root = Path(__file__).resolve().parents[2]
    if not (root / "src" / "oajoint" / "af3.py").exists():
        pytest.skip("project implementation not available")
    sys.path.insert(0, str(root / "src"))
    from oajoint import af3
    for f in ("profile_loglik", "em_fit", "em_fit_mix", "dose_slopes", "predictive_grid_mix"):
        import inspect
        assert inspect.getsource(getattr(af3, f)) == inspect.getsource(getattr(core, f)), f


def test_benchmark_on_synthetic_trials():
    from afstd import benchmark as bm
    arms, trials, drugs, events, _ = synthetic(n_trials=320, seed=7)
    ids = sorted(trials.trial_id); cut = ids[int(.7 * len(ids))]
    tr_ids = set(i for i in ids if i < cut); te_ids = set(ids) - tr_ids
    rec_tr = afstd.Records.from_frames(arms[arms.trial_id.isin(tr_ids)], trials, drugs, events)
    rec_te = afstd.Records.from_frames(arms[arms.trial_id.isin(te_ids)], trials, drugs, events)
    model = afstd.fit(rec_tr, version="3.2", outer=2, inner=4)
    Kt, q = afstd.predict_trial(model, rec_te)
    Kt = Kt.reset_index(drop=True); Kt["kk"] = np.arange(len(Kt))
    at = rec_te.arms.merge(Kt[["k", "kk"]], on="k"); at["off"] = 0.0
    soc_of_term = dict(zip(rec_tr.cf.term, rec_tr.cf.soc))
    pp = bm.placebo_prior(rec_tr.K, soc_of_term)
    mu, sd = pp(Kt)
    p0n = bm.placebo_nodes(Kt, mu, sd)
    null = np.zeros_like(q); null[:, np.argmin(np.abs(core.GRID))] = 1
    res = bm.score_all({"AF 3.2": q, "null": null}, {"AF 3.2": "hz", "null": "rr"}, Kt, at, np.random.default_rng(1), p0n=p0n)
    s32, pit = res["AF 3.2"]; s0, _ = res["null"]
    assert np.nanmean(s32) > np.nanmean(s0)
    cov = np.mean((pit >= .05) & (pit <= .95))
    assert 0.8 <= cov <= 0.98, cov


def test_conformance_vectors():
    import json
    d = Path(__file__).resolve().parents[1] / "conformance"
    arms = pd.read_csv(d / "arms.csv"); trials = pd.read_csv(d / "trials.csv"); events = pd.read_csv(d / "events.csv")
    drugs = pd.read_csv(d / "drugs.csv"); drugs["atc"] = drugs.atc.map(json.loads); drugs["targets"] = drugs.targets.map(json.loads)
    arms["drug_id"] = arms.drug_id.where(arms.drug_id.notna(), None)
    ref = pd.read_csv(d / "reference_estimates.csv")
    for v in ("3.0", "3.2", "3.3"):
        m = afstd.fit(afstd.Records.from_frames(arms, trials, drugs, events), version=v, outer=2, inner=6)
        e = afstd.estimates(m).merge(ref[ref.version == float(v)], on=["drug", "event"], suffixes=("", "_ref"))
        assert len(e) == (ref.version == float(v)).sum()
        for c, tol in (("psi_mean", 1e-4), ("psi_sd", 1e-4), ("af_hz", 1e-4), ("p_af_lt_half", 1e-3)):
            assert (e[c] - e[c + "_ref"]).abs().max() < tol, (v, c)


def test_af33_no_effect_probability(data):
    arms, trials, drugs, events, truth = data
    rec = afstd.Records.from_frames(arms, trials, drugs, events)
    est = afstd.estimates(afstd.fit(rec, version="3.3", outer=2, inner=4)).merge(truth, on=["drug", "event"])
    rich = est[est.n_trials >= 2]
    assert rich[rich.psi_true == 0].p_no_effect.mean() > rich[rich.psi_true != 0].p_no_effect.mean() + 0.1
