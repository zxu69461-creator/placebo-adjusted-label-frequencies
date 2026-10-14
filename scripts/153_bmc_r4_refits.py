#!/usr/bin/env python3
"""Refits for the third internal review (rules in 152_bmc_r4_prereg.py). Usage:
    153_bmc_r4_refits.py full | ds:<k>:<draw> | pivotal | addon
Settings as 127_jno_rev_refits.py (AF 3.3: em_fit_mix, outer=2, inner=6, spike); only the records change.
'full' refits the primary specification and also writes, from its fitted components, the prior-only probabilities
(variant bmc_prior_only) and the posteriors with the zero-excess weights scaled by c = 0, 0.5, 1.5 (bmc_pi0x<c>).
One row per listed pair and variant goes to jno_cell_variant; record/trial counts to jno_summary (bmc_r4_refit_info)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import af3  # noqa: E402
from oajoint import provenance as pv  # noqa: E402

COLS = ["drug", "term", "n_trials", "psi_mean", "psi_sd", "af_hz", "p_af_lt_half", "p_no_effect", "p_no_excess"]
LT = af3.GRID < np.log(2)


def approval_mask(con, K):
    appr = dict(con.execute("SELECT chembl_id, first_approval FROM ref_chembl_drug WHERE first_approval IS NOT NULL")
                .fetchall())
    yr = K.drug.map(appr); sy = K.start.dt.year
    return (yr.notna() & sy.notna() & (sy <= yr)).values


def indication_mask(con, K):
    ind = con.execute("""SELECT molecule_chembl_id, parent_molecule_chembl_id, lower(mesh_heading)
                         FROM ref_chembl_indication WHERE mesh_heading IS NOT NULL""").fetchall()
    app = {}
    for mol, par, h in ind:
        for d_ in {mol, par} - {None}:
            app.setdefault(d_, set()).add(h)
    cond = {}
    for n, meshes, anc in con.execute("SELECT nct, meshes, ancestors FROM ctgov_condition").fetchall():
        cond.setdefault(n, set()).update(x.lower() for x in (meshes or []) + (anc or []))
    return np.array([bool(app.get(d_, set()) & cond.get(n, set())) for n, d_ in zip(K.nct, K.drug)])


def components(fit, P, listed):
    """Prior-only P(AF < 0.5) and pi0-scaled posteriors for the listed pairs, from the fitted components."""
    cf = P["cf"]; post = fit["post"].astype(np.float64); G = len(af3.GRID)
    z0 = int(np.argmin(np.abs(af3.GRID)))
    sel = cf.reset_index().merge(listed, on=["drug", "term"])["index"].values
    soc = cf.s.values[sel]; pi0 = fit["pi0"][soc]
    shm = np.rint(fit["m"][sel] / af3.DZ).astype(np.int64)
    pidx = np.arange(G)[None, :] - shm[:, None]
    valid = (pidx >= 0) & (pidx < G)
    slab = np.where(valid, fit["gprior"][np.clip(pidx, 0, G - 1)], 0.0)
    slab /= slab.sum(1, keepdims=True)
    prior = (1 - pi0)[:, None] * slab; prior[:, z0] += pi0
    ps = post[sel]
    lik = np.where(prior > 0, ps / np.maximum(prior, 1e-300), 0.0)
    base = cf.loc[sel, ["drug", "term"]].reset_index(drop=True)
    out = {"bmc_prior_only": base.assign(p_af_lt_half=prior[:, LT].sum(1), p_no_effect=pi0)}
    for c in (0.0, 0.5, 1.0, 1.5):
        p0 = np.minimum(pi0 * c, 0.95)
        pr = (1 - p0)[:, None] * slab; pr[:, z0] += p0
        q = pr * lik; q /= np.maximum(q.sum(1, keepdims=True), 1e-300)
        sp = np.where(prior[:, z0] > 0, q[:, z0] * p0 / np.maximum(pr[:, z0], 1e-300), 0.0)
        psi = (q * af3.GRID).sum(1)
        out[f"bmc_pi0x{c:g}"] = base.assign(p_af_lt_half=q[:, LT].sum(1), p_no_effect=sp, psi_mean=psi,
                                            psi_sd=np.sqrt(np.maximum((q * af3.GRID ** 2).sum(1) - psi ** 2, 0)),
                                            af_hz=(q * (1 - np.exp(-af3.GRID))).sum(1),
                                            p_no_excess=q[:, af3.GRID <= 0].sum(1))
    check = float(np.abs(out["bmc_pi0x1"].p_af_lt_half.values - ps[:, LT].sum(1)).max())
    assert check < 1e-6, check                                     # c = 1 reproduces the fitted posterior
    del out["bmc_pi0x1"]
    return out


def main() -> int:
    variant = sys.argv[1]
    info = {}
    with pv.connect(read_only=True) as con:
        arms, K, J = af3.build(con)
        listed = con.execute("SELECT drug, term FROM afstd_pair WHERE on_fda").df()
        if variant == "full":
            P = af3.prepare(con, arms, K)
        elif variant.startswith("ds:"):
            _, k, draw = variant.split(":"); k, draw = int(k), int(draw)
            dense = con.execute("SELECT drug, term FROM afstd_pair WHERE on_fda AND evidence = '>=10'").df()
            rng = np.random.default_rng(20261014 + 10 * k + draw)
            Kd = K.merge(dense.rename(columns={"term": "t"}).assign(_d=1), on=["drug", "t"], how="left")
            drop = []
            for (_, _), g in Kd[Kd._d == 1].groupby(["drug", "t"], sort=True):
                tr = np.array(sorted(g.nct.unique()))
                keep = set(rng.choice(tr, min(k, len(tr)), replace=False))
                drop += list(g.k[~g.nct.isin(keep)])
            Ks = K[~K.k.isin(drop)]
            info = dict(dense_pairs=len(dense), dropped_records=len(drop))
            P = af3.prepare(con, arms[arms.k.isin(Ks.k)], Ks)
        elif variant == "pivotal":
            ph = dict(con.execute("SELECT nct, phase FROM ctgov_trial").fetchall())
            mk = dict(con.execute("SELECT nct, masking FROM ctgov_trial").fetchall())
            p34 = K.nct.map(lambda n: ph.get(n) is not None and any(x in ("PHASE3", "PHASE4") for x in ph[n])).values
            dbl = K.nct.map(lambda n: mk.get(n) in ("DOUBLE", "TRIPLE", "QUADRUPLE")).values
            m = p34 & dbl & indication_mask(con, K) & approval_mask(con, K)
            Ks = K[m]
            info = dict(p34=float(p34.mean()), double=float(dbl.mean()), records_kept=int(m.sum()))
            P = af3.prepare(con, arms[arms.k.isin(Ks.k)], Ks)
        elif variant == "addon":
            ac = con.execute("SELECT nct, eg_placebo FROM atlas_contrast").df().explode("eg_placebo")
            eg = con.execute("SELECT nct, eg_id, chembl_ids FROM ctgov_event_group").df()
            eg["mapped"] = eg.chembl_ids.map(lambda x: x is not None and len(x) > 0)
            pl = ac.merge(eg, left_on=["nct", "eg_placebo"], right_on=["nct", "eg_id"])
            addon = set(pl[pl.mapped].nct)
            inc = set(K.nct)
            info = dict(trials=len(inc), addon_trials=len(inc & addon), addon_share=len(inc & addon) / len(inc))
            P = af3.prepare(con, arms, K, trials=inc - addon)
        else:
            raise SystemExit(f"unknown variant {variant}")
    name = "bmc_" + variant.replace(":", "_")
    info.update(records=len(P["K"]), trials_fit=int(P["K"].nct.nunique()), pairs=len(P["cf"]))
    print(f"{name}: {info}", flush=True)
    fit, S, _, _ = af3.run_fit(P, fit_fn=af3.em_fit_mix, outer=2, inner=6, spike=True, verbose=False)
    S["p_no_effect"] = fit["p_null"]
    S["n_trials"] = S.n_trials.fillna(0).astype(int)
    out = {name: S[COLS].merge(listed, on=["drug", "term"])}
    if variant == "full":
        for v, d in components(fit, P, listed).items():
            d = d.merge(out[name][["drug", "term", "n_trials"]], on=["drug", "term"])
            for c in COLS:
                if c not in d:
                    d[c] = np.nan
            out[v] = d[COLS]
    for v, d in out.items():
        print(f"{v}: listed pairs {len(d):,}; expected share AF<0.5 {d.p_af_lt_half.mean():.4f}", flush=True)
    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        with pv.connect(read_only=True) as con:
            up = [r[0] for r in con.execute("SELECT out_id FROM prov_output_asset WHERE object_name IN "
                                            "('afstd_pair','afstd_label_match','atlas_contrast','ctgov_event_group',"
                                            "'ctgov_trial','ref_chembl_indication')").fetchall()]
        frames = []
        for v, d in out.items():
            d = d.copy(); d["run_id"] = run.run_id; d["variant"] = v
            frames.append(d[["run_id", "variant"] + COLS])
        with pv.connect() as con:
            for v in out:
                con.execute("DELETE FROM jno_cell_variant WHERE variant = ?", [v])
            con.execute("DELETE FROM jno_summary WHERE analysis = 'bmc_r4_refit_info' AND set_name = ?", [name])
        run.archive_table("jno_cell_variant", pd.concat(frames, ignore_index=True), upstream=up)
        R = pd.DataFrame([dict(run_id=run.run_id, analysis="bmc_r4_refit_info", set_name=name, grp=k, n_pairs=0,
                               n_drugs=0, metric=k, value=float(v), lo=np.nan, hi=np.nan) for k, v in info.items()])
        run.archive_table("jno_summary", R, upstream=up)
    return 0


if __name__ == "__main__":
    sys.exit(main())
