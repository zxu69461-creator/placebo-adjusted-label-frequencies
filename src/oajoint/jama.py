"""JAMA Network Open revision: display names, figure list and wording rules shared by the figure, table, manuscript and
supplement scripts (123-126). Descriptive, non-causal names only (JAMA: causal language only for single randomized
trials; a pooled analysis of trials is described in terms of association)."""
import re

FIG_PROFILE = "jama"

METHOD = {
    "AF 3.3": "Hierarchical model (this study)",
    "AF 3.2": "Model without zero-excess component",
    "B7 class-Bayes": "Class-informed Bayesian MA",
    "B3 Bayes-NN": "Bayesian random-effects MA",
    "B4 AF 2.0": "Empirical Bayes shrinkage",
    "B6 class-mean": "Drug-class average",
    "B2 PM-t": "Random-effects MA",
    "B1 MH": "Mantel-Haenszel pooling",
    "B5 term-only": "Adverse event average",
    "B0 null": "No difference from placebo",
}
COMPARATORS = ["AF 3.2", "B7 class-Bayes", "B3 Bayes-NN", "B4 AF 2.0", "B6 class-mean", "B2 PM-t", "B1 MH",
               "B5 term-only", "B0 null"]

MAIN_FIGS = [("jama_fig1_flow", "Figure 1"), ("jama_fig2_validation", "Figure 2"), ("jama_fig3_primary", "Figure 3")]
EFIGS = [("jama_efig1_calibration", "eFigure 1"), ("jama_efig2_discontinuation", "eFigure 2"),
         ("jama_efig3_frequencies", "eFigure 3"), ("jama_efig4_classes", "eFigure 4")]

# words that imply causation and must not describe our estimates in the JAMA text (allowed only when quoting the
# established term for a single trial or in the AI statement); checked by 125_jno_manuscript.py
CAUSAL_RE = re.compile(r"\b(caused by|causes|causing|attributable|attributed to the drug|effect of the drug|drug effect"
                       r"|because of the drug|due to the drug)\b", re.I)


def fmt_n(n: float) -> str:
    """JAMA digit style: 4-digit numbers closed up, 5 or more digits grouped with a thin space (eg, 45 380)."""
    s = f"{int(round(n)):d}"
    if len(s) <= 4:
        return s
    out = []
    while s:
        out.insert(0, s[-3:]); s = s[:-3]
    return " ".join(out)


def pct(x: float, d: int = 0) -> str:
    return f"{100 * x:.{d}f}%"


def ci(lo: float, hi: float, d: int = 0, unit: str = "%") -> str:
    """JAMA CI range: hyphen between non-negative values, 'to' when either limit is negative."""
    if unit == "%":
        a, b = f"{100 * lo:.{d}f}", f"{100 * hi:.{d}f}"
    else:
        a, b = f"{lo:.{d}f}", f"{hi:.{d}f}"
    if lo < 0 or hi < 0:
        return f"{a}{unit} to {b}{unit}"
    return f"{a}{unit}-{b}{unit}"
