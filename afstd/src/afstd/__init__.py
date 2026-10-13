"""afstd: reference implementation of the AF-STD specification.

    from afstd import Records, fit, estimates, predict_trial
    rec = Records.from_frames(arms, trials, drugs, events)
    model = fit(rec)                      # AF 3.x empirical-Bayes fit
    table = estimates(model)              # one row per drug x event
    q = predict_trial(model, new_trial_records)

See docs/AF-STD_spec for estimands, input schema and validation protocol."""
from .schema import Records
from .api import fit, estimates, predict_trial, report, SPEC_VERSION

__all__ = ["Records", "fit", "estimates", "predict_trial", "report", "SPEC_VERSION"]
