# Placebo-adjusted frequencies of adverse reactions listed on US drug labels

Analysis code for the manuscript *"Hierarchical modelling of randomized trials quantifies placebo-attributable background across prescription drug labels"* (submitted to *PLOS Medicine*).

For every drug–adverse event pair with placebo-controlled evidence on ClinicalTrials.gov, the code estimates the
excess fraction (the proportion of reports with the drug in excess of the placebo frequency) and frequencies per 100
patients with a hierarchical model of trial arms, validates the model by predicting trials that started after it was
fitted, and applies it to the adverse reactions listed on current US Food and Drug Administration labels.

## Contents

| Path | What it is |
|---|---|
| `afstd/` | Stand-alone reference implementation of the hierarchical model and the prediction benchmark, with tests and conformance data. Start here to inspect or reuse the method. |
| `src/oajoint/` | Shared modules: `af3.py` (hierarchical model), `bench.py` (prediction benchmark and scoring), `af2.py` (empirical Bayes comparator), `af_common.py`, `routes.py`, `provenance.py` (the only write path into the study database), `manifest.py` (audit), `jama.py`, `paper.py`, `paper_v3.py`, `figures.py`, `schematics.py`, `schematics_v3.py`. |
| `scripts/` | The analysis in run order (file-name numbers). |
| `db/schema.sql` | Schema of the study database (DuckDB). |

### Pipeline

| Scripts | Step |
|---|---|
| 32–34, 37, 64, 91 | Download ClinicalTrials.gov results (API v2), map drugs to ChEMBL, parse arms, adverse event groups, baseline and participant-flow modules, build placebo-controlled comparisons |
| 41–68, 73–75 | Conventional (Mantel–Haenszel) estimates and linkage to current FDA labels (openFDA) and 2016 labels (SIDER) |
| 80–83 | Empirical Bayes shrinkage (a comparator) |
| 92–102, 106 | Hierarchical model, simulations with known truth, prediction benchmark (development windows and temporal validation set) |
| 108–110c | Label analyses: expected proportions, organ systems, drug classes, frequencies per 100 patients, listing rules, sensitivity analyses |
| 115–122 | External benchmarks (statin individual participant data, ACE inhibitor cough), complete-reporting refit, label subsections, discontinuation, trial characteristics |
| 126–130b, 133–136 | Analyses added after internal review: refits (label-aware prior, trial subsets, half-sample and bootstrap refits), simulations, temporal calibration, positive controls, label audit, absolute frequencies without the posting threshold, formulation-matched subset |
| 137–142 | Labeled use (phase 3 or 4 trials, approved indications, approval timing, highest dose), label-linkage recall, and the table of estimates for every listed reaction (S1 Data) |
| 152–154 | Analyses added after a third internal review: deletion of trials of data-rich reactions, prior-only probabilities and zero-excess weight, pivotal-like trials, add-on designs, disclaimer lists, coverage, benchmarks, absolute excess |
| 123, 132, 151, 160, 163 | Figures of the manuscript and of its supporting information (160 and 163 draw the submitted versions) |

Scripts whose names end in `prereg` record analysis rules (with thresholds and pass criteria) in the decision table
`prov_decision` before the script that evaluates them is run; `python -m oajoint.manifest --verify` checks that every
evaluated rule was registered first. Internal identifiers map to the paper as follows: `AF 3.3` is the hierarchical
model, `AF 3.2` the model without its zero-excess component, `AF 1.0` conventional pooling, `B0`–`B7` the comparators
of Fig 2, and `bench22_confirm33` the evaluation in the temporal validation set.

## Requirements

Python 3.14 with the packages in `requirements.txt` (tested with the pinned versions). The analysis runs on a laptop
(11 cores, 18 GB RAM); each hierarchical fit takes minutes.

## Reproducing the analysis

```bash
pip install -r requirements.txt
export PYTHONPATH=src
python -m oajoint.provenance --init          # creates db/oajoint.duckdb from db/schema.sql
python scripts/32_ctgov_download.py          # then the remaining scripts in numerical order
```

All inputs are public and downloaded by the scripts: ClinicalTrials.gov (API v2), the openFDA drug-label export,
SIDER 4.1 and ChEMBL (including its drug indications). The paper used the downloads of September 24, 2026; later
downloads will include newer trials and labels. Intermediate files cached by scripts 98, 100 and 106 go to
`$OAJOINT_SCRATCH` (default `/tmp`).

Figure scripts (`111`, `112`, `123`, `132`, `151`, `160`, `163`, `src/oajoint/figures.py`, `schematics.py`) use `figkit`, a plotting helper
that is not part of this repository; all plotted values are in the study database and the supporting information of the paper.

### Reference implementation

```bash
pip install ./afstd
pytest afstd/tests
```

## Data and licences

Code: MIT licence (`LICENSE`). Input data remain under their providers' terms: ClinicalTrials.gov (terms of use of
the US National Library of Medicine), openFDA (CC0 1.0), SIDER 4.1 (CC BY-SA 4.0) and ChEMBL (CC BY-SA 3.0). Estimates
for every listed adverse reaction are provided with the paper (S1 Data).

## Citation

If you use this code, please cite the paper (citation to be added on publication) and this repository
(`CITATION.cff`).

Correspondence: Xingrong Guo and Xiaohu Zhu, Taihe Hospital, Hubei University of Medicine, Shiyan, China.
