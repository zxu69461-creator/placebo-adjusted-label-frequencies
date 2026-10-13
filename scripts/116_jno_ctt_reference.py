#!/usr/bin/env python3
"""Gate jno_ctt_reference: parse the CTT statin-label meta-analysis appendix into ext_ctt_outcome.

Source: Cholesterol Treatment Trialists' Collaboration. Assessment of adverse effects attributed to statin therapy in
product labels: a meta-analysis of double-blind randomised controlled trials. Lancet 2026;407:689-703 (PMID 41655587);
supplementary material from Europe PMC (PMC7619005), fetched with `rdata fetch` into raw/.
Webtable 1 gives the MedDRA preferred terms of each outcome; webfigures 1a-1d give statin versus placebo events, O-E,
Var(O-E), rate ratio and 95% CI, and mark FDR-significant outcomes with '*'.
"""
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import provenance as pv  # noqa: E402

ZIP = ROOT / "raw" / "CTT_Lancet2026_supplementary.zip"
PDF = "EMS212506-supplement-Supplementary_material.pdf"
SOCS = ["Blood and lymphatic system disorders", "Ear and labyrinth disorders", "Eye disorders",
        "Gastrointestinal disorders", "Hepatobiliary disorders", "Immune related disorders",
        "Metabolism and nutrition disorders", "Musculoskeletal and connective tissue disorders",
        "Nervous system disorders", "Psychiatric disorders", "Renal and urinary disorders",
        "Reproductive system and breast disorders", "Respiratory, thoracic and mediastinal disorders",
        "Skin and subcutaneous tissue disorders", "General disorders and administration site conditions"]
# webfigure names that are spelled differently in webtable 1
ALIAS = {"Non-specific muscle disorder": "Non-specific muscle disorder",
         "Asthenia, fatigue and malaise": "Asthenia, Fatigue, and malaise",
         "Vision impairment": "Visual impairment", "Thrombocytopaenia": "Thrombocytopenia",
         "RBC sedimentation rate increased": "Red blood cell sedimentation rate increased",
         "Dementia including Alzheimer's": "Dementia including Alzheimer's disease",
         "Other LFT abnormality": "Other liver function test abnormality"}
FDR_EXPECTED = {"Abnormal liver transaminases", "Other LFT abnormality", "Urinary composition alteration", "Oedema"}


def pdf_text(zpath):
    import fitz
    with zipfile.ZipFile(zpath) as z:
        data = z.read(PDF)
    doc = fitz.open(stream=data, filetype="pdf")
    return "\n".join(p.get_text("text") for p in doc)


def norm(s):
    return s.replace("−", "-").replace("’", "'")


def figures(txt):
    a, b = txt.index("Webfigure 1a:"), txt.index("Webfigure 2a:")
    L = [norm(x).strip() for x in txt[a:b].split("\n")]
    rr = re.compile(r"^(\d+\.\d+) \((\d+\.\d+) - (\d+\.\d+)\)$")
    cnt = re.compile(r"^(\d+) \((<?\d+\.\d+)\)$")
    num = re.compile(r"^-?\d+\.\d+$")
    soc, out = None, []
    for i, line in enumerate(L):
        if line.rstrip(" *") in SOCS:
            soc = line.rstrip(" *")
        m = rr.match(line)
        if not m:
            continue
        c1, c2 = cnt.match(L[i - 2]), cnt.match(L[i - 1])
        assert c1 and c2 and num.match(L[i - 4]) and num.match(L[i - 3]), L[i - 5:i + 1]
        raw = L[i - 5]
        out.append(dict(outcome=re.sub(r"[\s*†]+$", "", raw).replace(" *", "").strip(), soc=soc,
                        n_statin=int(c1.group(1)), n_placebo=int(c2.group(1)), oe=float(L[i - 4]),
                        var_oe=float(L[i - 3]), rr=float(m.group(1)), rr_lo=float(m.group(2)),
                        rr_hi=float(m.group(3)), fdr_sig="*" in raw))
    return out


def table_terms(txt, names):
    a, b = txt.index("Webtable 1:"), txt.index("Webfigure 1a:")
    wt = norm(txt[a:b])
    cut = wt.find("Events (% per annum)")
    wt = wt[:cut] if cut > 0 else wt

    def rx(name):
        return re.compile(r"(?m)^\s*" + r"\s+".join(re.escape(w) for w in name.split()) + r"\s*$")
    pos = []
    for nm in names:
        ms = list(rx(ALIAS.get(nm, nm)).finditer(wt))
        assert ms, f"outcome not found in webtable 1: {nm}"
        pos.append((ms[0].start(), ms[0].end(), "O", nm))
    for s in SOCS + ["Eosinophilia"]:
        pos += [(m.start(), m.end(), "S", s) for m in rx(s).finditer(wt)]
    pos.sort()
    terms = {}
    for i, (s, e, kind, nm) in enumerate(pos):
        if kind != "O":
            continue
        body = wt[e:pos[i + 1][0] if i + 1 < len(pos) else len(wt)]
        body = re.sub(r"=== PAGE \d+ ===|Page \d+ of \d+|System Organ\s+Class\s+CTT analysis outcome\s+description\s+"
                      r"Direct/close match included MedDRA Preferred Terms", " ", body)
        body = re.sub(r"-\s*\n\s*", "-", body)
        body = re.sub(r"\s+", " ", body).strip()
        terms[nm] = [t.strip() for t in body.split(";") if t.strip()]
    return terms


def main() -> int:
    txt = pdf_text(ZIP)
    rows = figures(txt)
    terms = table_terms(txt, [r["outcome"] for r in rows])
    for r in rows:
        r["pts"] = terms[r["outcome"]]
    df = pd.DataFrame(rows)
    fdr = set(df.loc[df.fdr_sig, "outcome"])
    ok = len(df) == 65 and bool((df.pts.map(len) > 0).all()) and fdr == FDR_EXPECTED
    print(f"outcomes {len(df)}; terms {df.pts.map(len).sum()}; FDR {sorted(fdr)}; pass {ok}")
    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        aid = run.register_input("ctt_lancet2026_supplement", asset_kind="publication_supplement", local_path=ZIP,
                                 accession="PMC7619005", source_registry="Europe PMC",
                                 url="https://www.ebi.ac.uk/europepmc/webservices/rest/PMC7619005/supplementaryFiles",
                                 genome_build="not_applicable", build_evidence="not genomic data",
                                 licence="Europe PMC author manuscript supplement (CC BY per Wellcome policy)",
                                 notes="CTT, Lancet 2026;407:689-703, PMID 41655587; webtable 1 + webfigures 1a-1d")
        run.evaluate_gate("jno_ctt_reference", float(len(df)), ok,
                          detail=dict(n_outcomes=len(df), n_terms=int(df.pts.map(len).sum()), fdr=sorted(fdr),
                                      soc_missing=int(df.soc.isna().sum())))
        if not ok:
            return 1
        df["run_id"] = run.run_id
        run.archive_table("ext_ctt_outcome", df[["run_id", "outcome", "soc", "pts", "n_statin", "n_placebo", "oe",
                                                 "var_oe", "rr", "rr_lo", "rr_hi", "fdr_sig"]],
                          inputs=[aid], mode="replace_all")
    return 0


if __name__ == "__main__":
    sys.exit(main())
