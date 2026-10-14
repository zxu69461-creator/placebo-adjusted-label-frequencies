#!/usr/bin/env python3
"""Figures of the BMC Medicine manuscript, read only from the project DuckDB, with BMC number style in every label
(commas for thousands, en-dash ranges, true minus signs; profile 'jama', 165 mm wide, within the BMC 170-mm width).

Fig. 1  study flow (123_jno_figures.fig1)
Fig. 2  frequencies of listed reactions with the drug and with placebo (new: scatter with the text's examples, excess
        per 100 in number-needed-to-harm bands, listed reactions with a large excess per drug)
Fig. 3  primary outcome (123_jno_figures.fig3 with the third-review sensitivity analyses added to panel C)
Fig. 4  validation (123_jno_figures.fig2)
Figures S1-S3 of Additional file 2: drug classes, discontinuation, temporal calibration (132_jno_efigures.py).
Outputs: figures/bmc/bmc_fig{1-4}_*.{pdf,png}, figures/bmc/bmc_efig{1,2,4}_*.{pdf,png}."""
import importlib.util
import re
import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import jama as J  # noqa: E402
from oajoint import provenance as pv  # noqa: E402
import figkit as fk  # noqa: E402

OUT = ROOT / "figures" / "bmc"


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def fmt_n(n: float) -> str:
    return f"{int(round(n)):,}"


def ci(lo: float, hi: float, d: int = 0, unit: str = "%") -> str:
    a, b = (f"{100 * lo:.{d}f}", f"{100 * hi:.{d}f}") if unit == "%" else (f"{lo:.{d}f}", f"{hi:.{d}f}")
    if lo < 0 or hi < 0:
        return f"{a}{unit} to {b}{unit}".replace("-", "\u2212")
    return f"{a}{unit}\u2013{b}{unit}"


def bmc_text(s: str) -> str:
    s = re.sub(r"(?<=\d)[\u2009\u202f ](?=\d{3}(?!\d))", ",", s)          # 47 828 -> 47,828
    s = re.sub(r"(?<=\d)(%?)-(?=\d)", "\\1\u2013", s)                      # 78.4%-81.3% -> 78.4%–81.3%; 2-4 -> 2–4
    s = re.sub(r"(?<![\w.])-(?=\d)", "\u2212", s)                          # -0.5 -> −0.5
    s = s.replace("Expected proportion", "Estimated proportion")       # wording of the text
    return s.replace("Mantel-Haenszel", "Mantel\u2013Haenszel")


class BMCFormatter(mpl.ticker.Formatter):
    """Tick labels are made at draw time: wrap each axis formatter (locations passed through) and restyle its output;
    integer tick labels of 1000 or more get a comma (no axis here shows years)."""

    def __init__(self, inner):
        self.inner = inner

    def set_axis(self, axis):
        super().set_axis(axis); self.inner.set_axis(axis)

    def set_locs(self, locs):
        super().set_locs(locs); self.inner.set_locs(locs)

    def get_offset(self):
        return self.inner.get_offset()

    def __call__(self, x, pos=None):
        s = bmc_text(self.inner(x, pos))
        return f"{int(s):,}" if re.fullmatch(r"\d{4,}", s) else s


def restyle(fig):
    for t in fig.findobj(mpl.text.Text):
        s = t.get_text()
        if s and bmc_text(s) != s:
            t.set_text(bmc_text(s))
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            for get, put in ((axis.get_major_formatter, axis.set_major_formatter),
                             (axis.get_minor_formatter, axis.set_minor_formatter)):
                f = get()
                if not isinstance(f, (BMCFormatter, mpl.ticker.NullFormatter)):
                    put(BMCFormatter(f))
    return fig


EXAMPLES = [("pregabalin", "dizziness"), ("pregabalin", "headache"), ("semaglutide", "nausea"),
            ("semaglutide", "headache"), ("metformin", "diarrhea"), ("metformin", "headache")]
ZONES = [("P(drug < 2× placebo) >97.5%", "blue"), ("2.5%-97.5%", "grey"), ("<2.5%", "vermilion")]


def fig2(d):
    """Frequencies with the drug and with placebo for every listed reaction; where the excess lies."""
    L = d["pair"]
    fig = fk.figure("double", height=115, profile=J.FIG_PROFILE)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1], height_ratios=[1, 1])
    # A -- scatter
    ax = fig.add_subplot(gs[:, 0])
    S_ = L[(L.per100_pl > 0) & (L.per100_drug_exp > 0)]
    masks = [S_.p_af_lt_half > .975, (S_.p_af_lt_half <= .975) & (S_.p_af_lt_half >= .025), S_.p_af_lt_half < .025]
    for (lab, col), m in zip(ZONES, masks):
        ax.scatter(S_.per100_pl[m], S_.per100_drug_exp[m], s=2.2, color=fk.c[col], alpha=.4, lw=0, rasterized=True,
                   label=f"{lab} ({fmt_n(m.sum())})")
    x = np.array([.01, 100])
    ax.plot(x, x, color=fk.c.ink, lw=.7, ls="--", label="Drug = placebo")
    ax.plot(x, 2 * x, color=fk.c.ink, lw=.7, ls=":", label="Drug = 2 × placebo")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(.03, 100); ax.set_ylim(.03, 100)
    for a_ in (ax.xaxis, ax.yaxis):
        a_.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_yticks([.1, 1, 10])                          # no label at the top end, clear of the panel letter
    ax.set_xlabel("Frequency with placebo per 100 patients")
    ax.set_ylabel("Frequency with the drug per 100 patients")
    ex = d["examples"]
    ax.scatter(ex.per100_pl, ex.per100_drug_exp, s=11, facecolor="none", edgecolor=fk.c.ink, lw=.8, zorder=5)
    fk.callouts(ax, ex.per100_pl.values, ex.per100_drug_exp.values,
                [f"{n.capitalize()}: {t}" for n, t in zip(ex.display_name, ex.term)])
    ax.legend(loc="lower right", frameon=False, markerscale=2.5)
    fk.panel(ax, "a", "Listed reactions")
    # B -- excess per 100 patients in bands
    ax2 = fig.add_subplot(gs[0, 1])
    a = L.per100_added_exp
    bands = [("<0", a < 0), ("0 to <1", (a >= 0) & (a < 1)), ("1 to <5", (a >= 1) & (a < 5)),
             ("5 to <10", (a >= 5) & (a < 10)), ("≥10", a >= 10)]
    sh = [100 * m.mean() for _, m in bands]
    cols = [fk.c.blue, fk.c.blue, fk.c.grey, fk.c.vermilion, fk.c.vermilion]
    ax2.bar(range(len(bands)), sh, color=cols, width=.7)
    for i, v in enumerate(sh):
        ax2.text(i, v + 1.2, f"{v:.1f}", ha="center", va="bottom", fontsize=fk.pt("annot"))
    ax2.set_xticks(range(len(bands))); ax2.set_xticklabels([b for b, _ in bands])
    top_ = 10 * np.ceil(max(sh) * 1.2 / 10)
    ax2.set_ylim(0, top_); ax2.set_yticks(np.arange(0, top_, 10)); ax2.spines["left"].set_bounds(0, top_ - 10)
    ax2.set_xlabel("Excess with the drug per 100 patients")
    ax2.set_ylabel("Listed reactions, %")
    fk.panel(ax2, "b", "Excess per 100 patients")
    d["fig2b"] = pd.DataFrame(dict(band=[b for b, _ in bands], share=sh))
    # C -- per drug: listed reactions with an excess of >= 5 per 100
    ax3 = fig.add_subplot(gs[1, 1])
    per = L.groupby("drug").agg(n=("term", "size"), ge5=("per100_added_exp", lambda x: int((x >= 5).sum())))
    per = per[per.n >= 5]
    k = per.ge5.clip(upper=8)
    cnt = k.value_counts().reindex(range(9), fill_value=0)
    ax3.bar(cnt.index, cnt.values, color=fk.c.blue, width=.7)
    med = float(per.ge5.median())
    ax3.axvline(med, color=fk.c.vermilion, lw=.8, ls="--", label=f"Median, {med:g}")
    ax3.set_xticks(range(9)); ax3.set_xticklabels([str(i) for i in range(8)] + ["≥8"])
    ax3.set_xlabel("Listed reactions with excess ≥5 per 100")
    ax3.set_ylabel(f"Drugs, No. (of {fmt_n(len(per))})")
    ax3.legend(loc="upper right", frameon=False)
    fk.panel(ax3, "c", "Per drug")
    d["fig2c"] = cnt
    return fig


def fig3(M123, d):
    """123_jno_figures.fig3 with the third-review refits added to the sensitivity panel (C)."""
    S0 = M123.S

    def J_(analysis, set_name, metric="exp_lt_half"):
        return S0(d, analysis, set_name, "all", metric, table="jsum")
    extra = [("Pivotal-like trials (refit)", J_("bmc_r4_pivotal", "listed")),
             ("No add-on designs (refit)", J_("bmc_r4_addon", "listed")),
             ("Disclaimer-only listings excluded", J_("bmc_r4_disclaimer", "excluding_disclaimer_only"))]
    fig = M123.fig3(d)
    # panel C (its rows are recorded in d["fig3c"]) is redrawn with the extra rows
    ax2 = [a for a in fig.axes if a.get_xlabel().startswith("Expected proportion") and len(a.get_yticks()) > 20][0]
    items = d["fig3c"].to_dict("records") + [dict(item=l, value=r.value, lo=r.lo, hi=r.hi, n=r.n_pairs) for l, r in extra]
    ax2.cla()
    yy = np.arange(len(items))[::-1]
    for yv, it in zip(yy, items):
        prim = it["item"].startswith("All listed")
        col = fk.c.vermilion if prim else fk.c.blue
        ax2.errorbar(it["value"], yv, xerr=[[it["value"] - it["lo"]], [it["hi"] - it["value"]]], fmt="D" if prim else "s",
                     ms=3.6, color=col, mec=col, elinewidth=1.0, capsize=1.5)
    ax2.axvline(.5, color=fk.c.ink, lw=.6, ls="--")
    ax2.axvline(items[0]["value"], color=fk.c.vermilion, lw=.5, ls=":")
    ax2.set_yticks(yy); ax2.set_yticklabels([i["item"] for i in items]); ax2.set_ylim(-.6, len(items) - .4)
    ax2.set_xlim(.47, .93); ax2.set_xticks([.5, .6, .7, .8, .9]); ax2.set_xticklabels(["50", "60", "70", "80", "90"])
    ax2.set_xlabel("Estimated proportion <0.5, % (95% CI)")
    fk.panel(ax2, "c", "Sensitivity")
    d["fig3c"] = pd.DataFrame(items)
    for a in fig.axes:                                   # wording of the text: "estimated proportion"
        if a.get_ylabel().startswith("Expected proportion"):
            a.set_ylabel(a.get_ylabel().replace("Expected", "Estimated"))
        if a.get_xlabel().startswith("Expected proportion"):
            a.set_xlabel(a.get_xlabel().replace("Expected", "Estimated"))
    for t in fig.findobj(mpl.text.Text):
        s_ = t.get_text()
        if s_.startswith("Expected proportion <0.5:"):
            t.set_text(s_.replace("Expected proportion", "Estimated proportion").replace("Posterior means", "Point estimates"))
    return fig


def main() -> int:
    J.fmt_n, J.ci = fmt_n, ci                     # the figure modules read J at call time
    M123 = _load("m123", "123_jno_figures.py")
    M132 = _load("m132", "132_jno_efigures.py")
    OUT.mkdir(exist_ok=True)
    made = {}
    with pv.connect(read_only=True) as con:
        d123 = M123.load(con)
        d132 = M132.load(con)
        d2 = dict(pair=con.execute("""SELECT p.drug, p.term, p.p_af_lt_half, p.per100_pl, e.per100_drug_exp,
                                         e.per100_added_exp FROM afstd_pair p JOIN afstd_pair_exp e USING (drug, term)
                                         WHERE p.on_fda ORDER BY drug, term""").df())
        nm = con.execute("SELECT DISTINCT chembl_id AS drug, display_name FROM ref_drug_name").df()
    ex = d2["pair"].merge(nm, on="drug")
    ex = pd.concat([ex[(ex.display_name == a) & (ex.term == b)] for a, b in EXAMPLES], ignore_index=True)
    assert len(ex) == len(EXAMPLES), ex
    d2["examples"] = ex
    figs = [("bmc_fig1_flow", lambda: M123.fig1(d123)), ("bmc_fig2_frequencies", lambda: fig2(d2)),
            ("bmc_fig3_primary", lambda: fig3(M123, d123)), ("bmc_fig4_validation", lambda: M123.fig2(d123)),
            ("bmc_efig1_classes", lambda: M132.efig4(d132)), ("bmc_efig2_discontinuation", lambda: M132.efig2(d132)),
            ("bmc_efig3_calibration", lambda: M132.efig1(d132))]
    for bid, fn in figs:
        made[bid] = fk.save(restyle(fn()), OUT / bid)
        print(bid, made[bid].report)
    with pv.run_context(__file__, env_name="oajoint-py311",
                        software={"matplotlib": mpl.__version__, "figkit": fk.version()}) as run:
        with pv.connect(read_only=True) as con:
            up = [r[0] for r in con.execute("""SELECT out_id FROM prov_output_asset WHERE object_name IN
                  ('afstd_pair','afstd_pair_exp','afstd_summary','jno_summary','bench_score','jno_ext_concordance',
                   'jno_label_section','ctgov_trial','atlas_contrast','ref_chembl_drug')""").fetchall()]
        for fid, saved in made.items():
            for path in saved.paths:
                run.archive_figure(fid, "all", path, panel_title=fid,
                                   source_objects=["afstd_pair", "afstd_summary", "jno_summary", "bench_score",
                                                   "jno_ext_concordance"], upstream=up)
            run.archive_figure_qa(fid, saved)
            run.archive_figure_abbrev(fid, saved.abbreviations)
    return 0


if __name__ == "__main__":
    sys.exit(main())
