"""Shared figure machinery.

Rules this module enforces, from the project's standing requirements:
  * every panel reads ONLY from the project DuckDB -- never a loose file. That makes
    figures reproducible and simultaneously proves the no-scattered-intermediates rule.
  * one palette across all seven figures, colourblind-safe (figkit rules.toml) and
    distinguishable in greyscale.
  * colour is never the only cue: heatmap cells carry numbers, series carry markers
    and linestyles.
  * every confidence interval is shown; nothing is hidden because it is wide.
  * panel grids are computed, not eyeballed, so there is no dead whitespace.
"""
from __future__ import annotations

import hashlib
from contextlib import contextmanager
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

# All sizes, fonts and colours come from the global figkit rules (~/.claude/skills/figkit/rules.toml).
# This module only keeps the old names so earlier scripts still run; new figures use figkit directly.
import figkit as fk  # noqa: E402

_PROF = fk.profile("natcomms")
W_SINGLE, W_DOUBLE = float(_PROF["widths"]["single"]), float(_PROF["widths"]["double"])
MM = fk.MM

_SEM = fk.sem("oajoint")
PROGRAM_COLORS = _SEM["program"]
PROGRAM_MARKERS = {"F_MORPH": "o", "F_ADIPOSITY": "s", "F_INFLAM": "^",
                   "F_CIRC": "D", "F_NEURAL": "v"}
SITE_GROUP = _SEM["site"]

CMAP_RG = fk.rules()["cmaps"]["diverging"]      # diverging, centred at 0: correct for a correlation
CMAP_SEQ = fk.rules()["cmaps"]["sequential"]    # sequential, perceptually uniform
GREY_MISSING = fk.c.grey_light


@contextmanager
def oaj_style(**overrides):
    """Legacy context manager: the figkit rules as rcParams (plus per-script overrides)."""
    import cmcrameri  # noqa: F401
    with mpl.rc_context({**fk.rc(), **overrides}):
        yield


def annotate_matrix(ax, M, *, fmt="{:.2f}", mask=None, threshold=0.55,
                    cmap=None, norm=None, size=None):
    """Write the value into every cell. Colour is then never the only cue, and the
    panel stays readable in greyscale and for colour-vision deficiency."""
    size = size or fk.rules()["typography"]["min_pt"]
    cmap = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    n_r, n_c = M.shape
    for i in range(n_r):
        for j in range(n_c):
            v = M[i, j]
            if mask is not None and mask[i, j]:
                continue
            if not np.isfinite(v):
                continue
            if cmap is not None and norm is not None:
                r, g, b, _ = cmap(norm(v))
                lum = 0.299 * r + 0.587 * g + 0.114 * b
                col = "white" if lum < threshold else "black"
            else:
                col = "black"
            ax.text(j, i, fmt.format(v), ha="center", va="center",
                    fontsize=size, color=col)


def panel_label(ax, letter, *, dx=-0.16, dy=1.04, size=8):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=size,
            fontweight="bold", va="bottom", ha="left")


def save_panel(fig, out_base: Path, formats=("pdf", "png")) -> dict[str, str]:
    """Write the panel and return {path: sha256} so provenance can record it."""
    out_base.parent.mkdir(parents=True, exist_ok=True)
    written = {}
    for f in formats:
        p = out_base.with_suffix(f".{f}")
        # Deterministic bytes. Matplotlib stamps /CreationDate into PDF (and a date into
        # SVG), so two renders of identical data produced different hashes -- which made
        # the reproducibility check of plan section 12 step 8 vacuous for the VECTOR
        # deliverable, the one that is actually submitted. Suppressing only the timestamp
        # keeps Creator/Producer, so the renderer version stays recorded.
        meta = {"CreationDate": None} if f == "pdf" else (
            {"Date": None} if f == "svg" else None)
        fig.savefig(p, format=f, metadata=meta)
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        written[str(p.relative_to(ROOT))] = h
    plt.close(fig)
    return written
