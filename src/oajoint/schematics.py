"""Illustrated schematics of the paper (figkit.art: SVG -> vector PDF, composed 1:1 into the saved figure).

fig1a_art(): the lead image of Fig. 1 — the hierarchical estimator as one continuous, upright picture:
  left   the trials: registry and a stack of trial cards (placebo and drug arms, events x/n), with the likelihood
         x ~ Binomial(n, 1 − exp[−exp(α_j + ζ_k + γΔd)]);
  centre the model: trial effects ζ_k ~ N(ψ + x_k′β, τ²) under the drug–side-effect effect ψ, which borrows from
         drug, event, class and target, with the prior ψ ~ π0 δ0 + (1 − π0)[m + G];
  right  the outputs per pair, each written as its definition; the attributable fraction leads.
Key statements are set larger (7 pt, the journal maximum) than their definitions (5 pt); nothing is tilted and
nothing is decorative: every element is a quantity of the model.
"""
from __future__ import annotations

from figkit import art

W1A, H1A = 176.0, 43.0          # design size (mm) of the Fig. 1a band


def _people(A, x0, top, events, col):
    k = 0
    for row in range(2):
        for colk in range(3):
            hit = k < events
            c = (("grey_dark" if hit else "grey_light") if col == "grey" else ("vermilion" if hit else "vermilion@t0.6"))
            A.person(x0 + 2.3 + colk * 3.9, top + 8.4 + row * 4.4, 3.6, color=c)
            k += 1


def fig1a_art() -> art.Art:
    A = art.Art(W1A, H1A)
    A.ribbon([(0, 22), (60, 20), (120, 20), (176, 20)], 24, 24, c0="blue@t0.8", c1="vermilion@t0.8", opacity=0.25)

    # ---- the trials ----------------------------------------------------------------------------------
    A.cylinder(1.4, 6.6, 7.0, 9.6, color="blue")
    A.text(4.9, 19.4, "ClinicalTrials", role="fine", anchor="middle", color="grey_dark")
    A.text(4.9, 21.5, ".gov", role="fine", anchor="middle", color="grey_dark")
    A.arrow(9.4, 11.4, 13.8, 11.4, width=1.3, color="grey", taper=False)
    cx, cy, cw, ch = 14.6, 5.8, 30.0, 22.6
    A.stack(cx, cy, cw, ch, n=3, dx=1.6, dy=-1.6)
    top = A.card(cx, cy, cw, ch, header=("trial k", "blue"), shadow=False)
    for name, ax0, col, events, sub in (("placebo", cx + 1.2, "grey", 1, "0"),
                                        ("drug, dose d", cx + cw / 2 + 0.4, "vermilion", 4, "1")):
        A.text(ax0 + 6.3, top + 2.4, name, role="annot", anchor="middle",
               color="grey_dark" if col == "grey" else "vermilion", bold=True)
        _people(A, ax0, top, events, col)
        A.math(ax0 + 6.3, cy + ch - 1.9, f"x_{{{sub}}} / n_{{{sub}}}", role="annot", anchor="middle",
               color="grey_dark" if col == "grey" else "vermilion")
    A.line(cx + cw / 2, top + 1.0, cx + cw / 2, cy + ch - 1.0, color="grey_light")
    A.math(0.6, 33.4, "x ~ Binomial(n, 1 − exp[−exp(α_{j} + ζ_{k} + γΔd)])", role="legend")
    A.math(0.6, 36.9, "α_{j}: trial background, profiled out;  γ: dose slope", role="fine", color="grey_dark")
    A.text(0.6, 39.5, "a placebo arm shared by several drugs is split between them", role="fine", color="grey_dark")

    # ---- the model -----------------------------------------------------------------------------------
    A.arrow(50.0, 17.0, 60.6, 17.0, width=2.4, color="grey", taper=False)
    psi = (86.0, 12.4)
    zy, zs = 24.6, (73.6, 86.0, 98.4)
    for zx in zs:
        A.line(psi[0], psi[1] + 4.0, zx, zy - 2.3, color="grey", sw=A.lw_data)
    sats = [(76.6, 2.8, "indigo", "drug"), (72.4, 6.9, "teal", "event"),
            (70.8, 11.8, "purple", "drug class"), (72.4, 16.7, "sky", "target")]
    for sx, sy, col, lab in sats:
        A.curved_arrow(sx + 1.5, sy + (0.4 if sy < psi[1] else -0.4), psi[0] - 4.3,
                       psi[1] + (sy - psi[1]) * 0.25, bend=0.12 if sy < psi[1] else -0.12, color="grey", head=1.1)
    for sx, sy, col, lab in sats:
        A.sphere(sx, sy, 1.4, color=col)
        A.text(sx - 2.1, sy, lab, role="fine", anchor="end", color="grey_dark")
    A.sphere(*psi, 4.0, color="vermilion")
    A.text(psi[0], psi[1] + 0.2, "ψ", role="base", anchor="middle", color="white", bold=True)
    for k, zx in enumerate(zs):
        A.sphere(zx, zy, 2.3, color="blue")
        A.text(zx, zy + .1, "ζ_{" + ("1", "2", "k")[k] + "}", role="annot", anchor="middle", color="white", bold=True)
    A.math(86.0, 31.2, "ζ_{k} ~ N(ψ + x_{k}′β, τ^{2})", role="legend", anchor="middle")
    A.math(86.0, 34.6, "x_{k}: blinding, phase, sponsor, population, dose, follow-up", role="fine",
           anchor="middle", color="grey_dark")
    A.math(86.0, 37.4, "m: cross-fitted from drug, event, class and target;  π_{0}, τ per organ system",
           role="fine", anchor="middle", color="grey_dark")
    A.path(f"M{psi[0] + 4.1},{psi[1] - 0.8} L97.2,8.6", stroke="grey", dash="0.8 0.7")
    A.density(97.6, 1.6, 19.6, 8.8, spike_at=0.1, color="vermilion", spike="blue")
    A.math(100.2, 2.0, "π_{0}", role="fine", color="blue")
    A.math(113.2, 4.6, "m + G", role="fine", color="vermilion")
    A.math(99.6, 12.1, "0", role="fine", anchor="middle", color="grey_dark")
    A.text(117.2, 12.1, "log HR", role="fine", anchor="end", color="grey_dark")
    A.math(107.4, 16.4, "ψ ~ π_{0}δ_{0} + (1 − π_{0})[m + G]", role="legend", anchor="middle")

    # ---- the outputs per pair --------------------------------------------------------------------------
    A.arrow(102.4, 24.6, 125.2, 24.6, width=2.4, color="vermilion", taper=False)
    ox, oy, ow, oh = 126.4, 1.6, 49.2, 39.8
    A.card(ox, oy, ow, oh, r=1.4, shadow=True)
    A.line(ox + 3.0, oy + 11.0, ox + ow - 3.0, oy + 11.0, color="grey_light")
    A.math(ox + 3.0, oy + 4.6, "AF = 1 − exp(−ψ)", role="base", color="vermilion")
    A.text(ox + 3.0, oy + 8.1, "attributable fraction of the reports on the drug, 95% CrI", role="fine",
           color="grey_dark")
    rows = [("P(AF < 0.5)", "the entry is mostly background"),
            ("P(ψ = 0)", "the drug does not cause it"),
            ("100·p_{0},  100·(p_{1} − p_{0})", "per 100 patients: on placebo, added by the drug"),
            ("ψ + ζ_{new}", "95% interval for a new trial")]
    for k, (formula, gloss) in enumerate(rows):
        yy = oy + 14.9 + k * 6.3
        A.math(ox + 3.0, yy, formula, role="legend")
        A.text(ox + 3.0, yy + 2.6, gloss, role="fine", color="grey_dark")
    return A


# ================================================================================================ Fig. 5d
W5D = 52.0                      # design width (mm) of the Fig. 5d label illustration; height follows the rows
ROW5D = 5.4                     # mm per label entry


def h5d(n_rows: int) -> float:
    return 33.0 + (n_rows - 4) * ROW5D

VERDICT = {"confident_bg": "mostly background", "probable_bg": "probably background",
           "probable_drug": "probably the drug", "confident_drug": "mostly the drug"}


def fig5d_art(drug: str, rows: list[dict], zone_color: dict) -> art.Art:
    """One label entry per row: what the label says today (frequency on the drug, grey, right) and what the proposed
    line says (per 100 patients: on placebo + added by the drug with its 95% interval, and what that means).
    Real rows of one drug; rows: term, drug, placebo, added, lo, hi, zone."""
    H5D = h5d(len(rows))
    A = art.Art(W5D, H5D)
    top = A.card(0.4, 0.6, W5D - 0.8, H5D - 1.2, header=(f"{drug} · per 100 patients", "vermilion"), shadow=True)
    # every row the same: term and verdict on the first line; bar, its numbers and (right, grey) what the label
    # says today on the second
    scale = 22.0 / max(r["drug"] + max(r["hi"] - r["added"], 0) for r in rows)
    bx = 2.6
    for k, r in enumerate(rows):
        y = top + 2.6 + k * ROW5D
        A.text(bx, y, r["term"].capitalize(), role="legend", color="ink", bold=True)
        A.text(W5D - 2.4 - 1.1, y + 2.5, f"label {r['drug']:.0f}%", role="annot", anchor="end", color="grey")
        chip = VERDICT[r["zone"]]
        cw = art.MM_PER_PT * 5 * 0.56 * len(chip) + 2.2
        A.rect(W5D - 2.4 - cw, y - 1.35, cw, 2.7, r=1.3, fill=zone_color[r["zone"]])
        A.text(W5D - 2.4 - 1.1, y, chip, role="fine", anchor="end", color="white", bold=True)
        yb = y + 2.5
        pl, add = r["placebo"] * scale, max(r["added"], 0) * scale
        A.rect(bx, yb - 0.9, pl, 1.8, r=0.3, fill="grey_light")
        A.rect(bx + pl, yb - 0.9, add, 1.8, r=0.3, fill="vermilion")
        lo, hi = r["placebo"] + max(r["lo"], 0), r["placebo"] + r["hi"]
        A.line(bx + lo * scale, yb, bx + hi * scale, yb, color="ink", sw=A.lw)
        for q in (lo, hi):
            A.line(bx + q * scale, yb - 0.55, bx + q * scale, yb + 0.55, color="ink", sw=A.lw)
        A.text(bx + hi * scale + 1.2, yb,
               f"{r['placebo']:.0f} + {max(r['added'], 0):.0f} ({max(r['lo'], 0):.0f}–{r['hi']:.0f})",
               role="fine", color="grey_dark")
    ly = H5D - 2.6
    A.rect(bx, ly - 0.9, 1.8, 1.8, r=0.3, fill="grey_light")
    A.text(bx + 2.6, ly, "on placebo", role="fine", color="grey_dark")
    A.rect(bx + 14.0, ly - 0.9, 1.8, 1.8, r=0.3, fill="vermilion")
    A.text(bx + 16.6, ly, "added by the drug (95% interval)", role="fine", color="grey_dark")
    return A
