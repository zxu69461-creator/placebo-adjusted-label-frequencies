"""Reference test vectors for AF-STD conformance: a fixed synthetic dataset (seed 20260927) and the reference
implementation's estimates. An implementation conforms if it reproduces psi_mean, psi_sd and af_hz within 1e-4 and
p_af_lt_half within 1e-3 for every drug x event."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))
import afstd  # noqa: E402
from afstd.simulate import synthetic  # noqa: E402


def main():
    arms, trials, drugs, events, truth = synthetic(seed=20260927)
    out = HERE / "conformance"
    arms.to_csv(out / "arms.csv", index=False); trials.to_csv(out / "trials.csv", index=False)
    drugs.assign(atc=drugs.atc.map(json.dumps), targets=drugs.targets.map(json.dumps)).to_csv(out / "drugs.csv", index=False)
    events.to_csv(out / "events.csv", index=False)
    rows = []
    for v in ("3.0", "3.2", "3.3"):
        m = afstd.fit(afstd.Records.from_frames(arms, trials, drugs, events), version=v, outer=2, inner=6)
        e = afstd.estimates(m)[["drug", "event", "psi_mean", "psi_sd", "af_hz", "p_af_lt_half"]].assign(version=v)
        rows.append(e)
    import pandas as pd
    pd.concat(rows).to_csv(out / "reference_estimates.csv", index=False, float_format="%.8f")
    (out / "README.md").write_text("AF-STD conformance vectors: inputs (arms, trials, drugs, events; list columns JSON-encoded) and "
                                   "reference_estimates.csv from the reference implementation (AF 3.0, 3.2 and 3.3, outer=2, inner=6). "
                                   "Tolerances: psi_mean, psi_sd, af_hz 1e-4; p_af_lt_half 1e-3.\n")
    print("wrote conformance vectors")


if __name__ == "__main__":
    main()
