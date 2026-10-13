"""Generate src/afstd/core.py from the validated project implementation (oajoint.af3 / oajoint.af2), verbatim, so that
the reference package and the benchmarked estimator cannot drift apart. Records the SHA-256 of the source modules."""
import hashlib
import inspect
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent / "src"))
from oajoint import af2, af3, bench  # noqa: E402

FUNCS_AF3 = ["design_covariates", "rows_for", "_cll", "profile_loglik", "dose_slopes", "_kernels", "_shift_idx", "prior_mean",
             "em_fit", "em_fit_mix", "_add_rows", "baseline_terms", "summarise", "predictive_grid", "predictive_grid_mix",
             "prior_for_new", "design_multi", "term_tokens"]
FUNCS_AF2 = ["_design", "_ridge"]


def main():
    src3 = Path(af3.__file__).read_bytes(); src2 = Path(af2.__file__).read_bytes()
    head = f'''"""afstd.core -- VENDORED VERBATIM from the benchmarked project implementation. Do not edit by hand; regenerate
with tools/vendor_core.py. Source: oajoint/af3.py sha256 {hashlib.sha256(src3).hexdigest()[:16]}, oajoint/af2.py sha256
{hashlib.sha256(src2).hexdigest()[:16]}."""
import re

import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.sparse.linalg import lsqr

GRID = np.round(np.arange(-5.0, 5.0 + 1e-9, 0.05), 4)
DZ = 0.05
COVS = {af3.COVS!r}
FEATS = {af3.FEATS!r}
TAU_GRID = np.array({list(af3.TAU_GRID)!r})
_STOP = {sorted(af3._STOP)!r}
_STOP = set(_STOP)

'''
    body = []
    for f in FUNCS_AF2:
        body.append(inspect.getsource(getattr(af2, f)))
    body.append("\n\nclass af2:  # shim: vendored functions refer to af2._design / af2._ridge\n"
                "    _design = staticmethod(_design)\n    _ridge = staticmethod(_ridge)\n")
    for f in FUNCS_AF3:
        body.append(inspect.getsource(getattr(af3, f)))
    out = HERE / "src" / "afstd" / "core.py"
    out.write_text(head + "\n\n".join(body))
    print("wrote", out, len(out.read_text().splitlines()), "lines")
    # benchmark (BENCH-2.1): scoring rules, placebo background model, baselines
    srcb = Path(bench.__file__).read_bytes()
    hb = f'''"""afstd.benchmark -- BENCH-2.1, VENDORED VERBATIM from oajoint/bench.py (sha256 {hashlib.sha256(srcb).hexdigest()[:16]}).
Score any estimator: give each test record a predictive density of its log effect on core.GRID (scale 'rr' or 'hz'); the
placebo risk is integrated over the hierarchical background model; proper log score and PIT calibration."""
import numpy as np
import pandas as pd
from scipy import stats

from . import core as af3

BASE_COVS = {bench.BASE_COVS!r}

'''
    names = ["trial_records", "_pm_tau2", "baselines", "_grid_normal", "_grid_t", "baseline_predictives", "score_all",
             "placebo_prior", "placebo_nodes", "class_mean"]
    extra = [inspect.getsource(getattr(af3, f)) for f in ("beta_nodes", "_arm_p", "score_records_beta", "pit_records_beta")]
    (HERE / "src" / "afstd" / "core_extra.py").write_text(
        '"""Vendored scoring primitives (from oajoint/af3.py)."""\nimport numpy as np\nfrom .core import GRID, _add_rows\n\n\n' + "\n\n".join(extra))
    bb = [inspect.getsource(getattr(bench, f)) for f in names]
    txt = hb + "\n\n".join(bb)
    txt = txt.replace("from . import core as af3", "from . import core as af3\nfrom . import core_extra as _cx\nfor _n in ('beta_nodes', 'score_records_beta', 'pit_records_beta', '_arm_p'):\n    setattr(af3, _n, getattr(_cx, _n))")
    (HERE / "src" / "afstd" / "benchmark.py").write_text(txt)
    print("wrote benchmark.py")


if __name__ == "__main__":
    main()
