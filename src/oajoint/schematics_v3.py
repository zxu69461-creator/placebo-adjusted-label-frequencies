"""Illustrations of the PLOS Medicine manuscript (figkit.art; composed 1:1 into the saved figure).

fig1a_art(): the hierarchical model as a conceptual diagram for clinical readers (equations only in the Methods):
  left the trial arms and the arm-level count model, centre the hierarchical empirical-Bayes prior with its
  spike-and-slab shape, right the four outputs per drug-event pair. Labels 6.5 pt, output formulas 7 pt.
fig5d_art(): a proposed label block for one drug, with the four posterior zones of the paper as badges and the
  numbers in their own aligned columns.
"""
from __future__ import annotations

from figkit import art

from oajoint import paper_v3 as P3

W1A, H1A = 176.0, 46.0          # design size (mm) of the Fig. 1a band


def _people(A, x0, top, events, col):
    k = 0
    for row in range(2):
        for colk in range(3):
            hit = k < events
            c = (("grey_dark" if hit else "grey_light") if col == "grey" else ("vermilion" if hit else "vermilion@t0.6"))
            A.person(x0 + 2.3 + colk * 3.9, top + 8.0 + row * 4.3, 3.5, color=c)
            k += 1


def fig1a_art() -> art.Art:
    """Conceptual diagram for clinical readers (the equations are in the Methods): trial arms with the arm-level count
    model; the hierarchical empirical-Bayes prior with its spike-and-slab shape; the four outputs per drug-event pair.
    Labels are set at 6.5 pt (legend) and the output formulas at 7 pt (base)."""
    A = art.Art(W1A, H1A)
    G = "grey_dark"

    # ---- trial arms ------------------------------------------------------------------------------------
    A.db(1.6, 4.4, 7.2, 9.0, color="blue")
    for k, t in enumerate(("Clinical", "Trials", ".gov")):
        A.text(5.2, 16.6 + 2.1 * k, t, role="fine", anchor="middle", color=G)
    A.arrow(9.6, 10.4, 13.6, 10.4, width=1.3, color="grey", taper=False)
    cx, cy, cw, ch = 14.4, 3.2, 31.0, 21.8
    top = A.card(cx, cy, cw, ch, header=("trial j, record k", "blue"), shadow=False)
    for name, ax0, col, events, sub in (("placebo", cx + 1.0, "grey", 1, "0"),
                                        ("drug", cx + cw / 2 + 0.6, "vermilion", 4, "1")):
        A.text(ax0 + 6.4, top + 2.3, name, role="legend", anchor="middle",
               color=G if col == "grey" else "vermilion", bold=True)
        _people(A, ax0, top, events, col)
        A.math(ax0 + 6.4, cy + ch - 1.8, f"x_{{{sub}}} / n_{{{sub}}}", role="legend", anchor="middle",
               color=G if col == "grey" else "vermilion")
    A.line(cx + cw / 2, top + 1.0, cx + cw / 2, cy + ch - 1.0, color="grey_light")
    for k, (line, indent) in enumerate((("• Arm-level count model", 0), ("(baseline risk α profiled out)", 1),
                                        ("• Within-trial dose slope (γ);", 0), ("shared placebo split", 1))):
        A.text(1.0 + 2.4 * indent, 30.0 + 3.3 * k + (1.0 if k >= 2 else 0), line, role="legend", color="ink")

    # ---- hierarchical empirical-Bayes prior ---------------------------------------------------------------
    A.arrow(48.6, 13.6, 56.4, 13.6, width=2.2, color="grey", taper=False)
    psi = (84.0, 10.4)
    zy, zs = 21.6, (72.4, 84.0, 95.6)
    for zx in zs:
        A.line(psi[0], psi[1] + 3.6, zx, zy - 2.2, color="grey", sw=A.lw_data)
    sats = [(73.6, 2.4, "indigo", "drug"), (69.6, 6.2, "teal", "event"),
            (68.2, 10.4, "purple", "drug class"), (69.6, 14.6, "sky", "target")]
    for sx, sy, col, lab in sats:
        A.line(sx + 1.3, sy, psi[0] - 3.7, psi[1] + (sy - psi[1]) * 0.3, color="grey", sw=A.lw_data)
    for sx, sy, col, lab in sats:
        A.circle(sx, sy, 1.25, fill=col)
        A.text(sx - 2.0, sy, lab, role="legend", anchor="end", color=G)
    A.circle(*psi, 3.7, fill="vermilion")
    A.math(psi[0], psi[1] + 0.2, "ψ_{dt}", role="legend", anchor="middle", color="white")
    for k, zx in enumerate(zs):
        A.circle(zx, zy, 2.2, fill="blue")
        A.math(zx, zy + .1, "ζ_{" + ("1", "2", "k")[k] + "}", role="legend", anchor="middle", color="white")
    # spike-and-slab shape of the pair effect
    A.path(f"M{psi[0] + 3.8},{psi[1] - 0.6} L97.6,7.8", stroke="grey", dash="0.8 0.7")
    A.density(98.0, 1.0, 20.0, 8.6, spike_at=0.1, color="vermilion", spike="blue")
    A.math(100.8, 3.0, "π_{0s}", role="legend", color="blue")
    A.math(113.0, 3.6, "m_{dt} + G", role="legend", color="vermilion")
    A.math(100.0, 11.6, "0", role="annot", anchor="middle", color=G)
    A.text(118.0, 11.6, "log HR", role="annot", anchor="end", color=G)
    for k, (line, indent, math) in enumerate((("• Cross-fitted prior from drug, event,", 0, False),
                                              ("class & target", 1, False),
                                              ("• Adjusted for trial design (x_{k}) &", 0, True),
                                              ("between-trial heterogeneity (τ)", 1, False))):
        x, y = 60.0 + 2.4 * indent, 30.0 + 3.3 * k + (1.0 if k >= 2 else 0)
        (A.math if math else A.text)(x, y, line, role="legend", color="ink")

    # ---- outputs per drug-event pair --------------------------------------------------------------------
    A.arrow(104.0, 21.6, 126.0, 21.6, width=2.2, color="vermilion", taper=False)
    ox, oy, ow, oh = 127.4, 0.8, 48.2, 44.6
    A.card(ox, oy, ow, oh, r=1.4, shadow=False)
    items = [("AF = 1 − exp(−ψ)", "vermilion", ("Attributable fraction", "(share caused by drug)")),
             ("P(AF < 0.5)", "ink", ("Prob. background-dominant", "(relative risk < 2)")),
             ("P(ψ = 0)", "ink", ("Prob. zero excess risk", "(no drug effect)")),
             ("100·p_{0} and 100·(p_{1} − p_{0})", "ink", ("Events per 100 on placebo", "& added by drug"))]
    for k, (formula, col, (g1, g2)) in enumerate(items):
        yy = oy + 4.4 + k * 10.4
        A.math(ox + 2.6, yy, formula, role="base", color=col)
        A.text(ox + 2.6, yy + 3.5, g1, role="legend", color=G, italic=True)
        A.text(ox + 2.6, yy + 6.4, g2, role="legend", color=G, italic=True)
        if k < len(items) - 1:
            A.line(ox + 2.6, yy + 8.6, ox + ow - 2.6, yy + 8.6, color="grey_faint")
    return A


# ================================================================================================ Fig. 5d
W5D = 67.0                      # design width (mm) of the Fig. 5d label block; height follows the rows
ROW5D = 5.6                     # mm per label entry
HEAD5D = 12.6                   # card header + column headers
FOOT5D = 5.0


def h5d(n_rows: int) -> float:
    return HEAD5D + n_rows * ROW5D + FOOT5D


def fig5d_art(drug: str, rows: list[dict], zone_color: dict) -> art.Art:
    """One label entry per row, in aligned columns: term and bar (left); 'placebo + added (95% CrI)' per 100 patients
    (numbers, right-aligned); the frequency on the current label (grey); the posterior zone as a two-line badge.
    Real rows of one drug; rows: term, drug, placebo, added, lo, hi, zone."""
    H = h5d(len(rows))
    A = art.Art(W5D, H, kind="mockup")      # a label mock-up: the content is text (no head-figure budgets)
    top = A.card(0.4, 0.6, W5D - 0.8, H - 1.2, header=(f"{drug}: adverse reactions per 100 patients", "vermilion"),
                 shadow=False)
    bx, bw = 2.4, 14.5                       # bar column
    xn = 37.4                                # right edge of the numbers column
    xl = 43.4                                # right edge of the label column
    xb = 44.6                                # left edge of the badge column
    yh = top + 2.6
    A.text(bx, yh, "Reaction", role="fine", color="grey_dark", bold=True)
    A.text(xn, yh, "placebo + added (95% CrI)", role="fine", anchor="end", color="grey_dark", bold=True)
    A.text(xl, yh, "label", role="fine", anchor="end", color="grey_dark", bold=True)
    A.text(xb, yh, "zone", role="fine", color="grey_dark", bold=True)
    A.line(bx, yh + 1.7, W5D - 2.6, yh + 1.7, color="grey_light")
    scale = bw / max(r["placebo"] + max(r["hi"], 0) for r in rows)
    for k, r in enumerate(rows):
        y = yh + 4.4 + k * ROW5D
        A.text(bx, y, r["term"].capitalize(), role="annot", color="ink", bold=True)
        yb = y + 2.7
        pl, add = r["placebo"] * scale, max(r["added"], 0) * scale
        A.rect(bx, yb - 0.85, pl, 1.7, r=0.3, fill="grey_light")
        A.rect(bx + pl, yb - 0.85, add, 1.7, r=0.3, fill="vermilion")
        lo, hi = r["placebo"] + max(r["lo"], 0), r["placebo"] + max(r["hi"], 0)
        A.line(bx + lo * scale, yb, bx + hi * scale, yb, color="ink", sw=A.lw)
        for q in (lo, hi):
            A.line(bx + q * scale, yb - 0.5, bx + q * scale, yb + 0.5, color="ink", sw=A.lw)
        ym = y + 1.35
        A.text(xn, ym, f"{r['placebo']:.0f} + {max(r['added'], 0):.0f} ({max(r['lo'], 0):.0f}–{max(r['hi'], 0):.0f})",
               role="annot", anchor="end", color="ink")
        A.text(xl, ym, f"{r['drug']:.0f}%", role="annot", anchor="end", color="grey")
        l1, l2 = P3.ZONE_LINES[r["zone"]]
        A.rect(xb, y - 1.05, W5D - 2.0 - xb, 4.7, r=1.0, fill=zone_color[r["zone"]])
        txt = "white" if r["zone"] in ("confident_bg", "confident_drug") else "ink"
        A.text(xb + 0.9, y + 0.1, l1, role="fine", color=txt, bold=True)
        A.text(xb + 0.9, y + 2.4, l2, role="fine", color=txt, bold=True)
        if k < len(rows) - 1:
            A.line(bx, y + 4.25, W5D - 2.0, y + 4.25, color="grey_faint")
    ly = H - 3.0
    A.rect(bx, ly - 0.85, 1.7, 1.7, r=0.3, fill="grey_light")
    A.text(bx + 2.5, ly, "on placebo", role="fine", color="grey_dark")
    A.rect(bx + 15.0, ly - 0.85, 1.7, 1.7, r=0.3, fill="vermilion")
    A.text(bx + 17.5, ly, "added by the drug (95% credible interval)", role="fine", color="grey_dark")
    return A
