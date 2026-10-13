#!/usr/bin/env python3
"""Parse the ClinicalTrials.gov corpus into auditable trial / event-group / term tables.

Three problems have to be solved here, and each is solved in a way that leaves a trace:

1. EVENT GROUPS ARE NOT ARMS. The adverse-event module reports counts by "event group"
   (EG000, EG001, ...), which carry their own titles and share NO identifier with the
   protocol's arm groups. Each event group is linked to an arm by title similarity; the
   method and score are stored per row, so a weak link is visible, never assumed.

2. PLACEBO VS ACTIVE. Role comes from the linked arm's type (PLACEBO_COMPARATOR, SHAM_
   COMPARATOR -> placebo; EXPERIMENTAL, ACTIVE_COMPARATOR -> active). Only when no arm
   links does it fall back to the event-group title, and then only if the title says
   placebo AND no drug ingredient can be read from it -- so a double-dummy arm such as
   "Drug A + placebo for Drug B" is not misread as placebo.

3. DRUG NAME -> CLASS. Intervention names are free text ("Semaglutide 2.4 mg s.c.",
   "LY2963016"). They are normalised (dose, route and formulation words removed) and
   matched to ChEMBL synonyms: exact name, then the trial's own MeSH terms, then parts of
   combination names, then single tokens of >= 5 characters. The method used is stored.

Every term-level row is kept (Parquet domain ctgov_ae), and the analyses read all adverse-event
outcomes from it. Summary columns computed here for a separate serious-event term set are not used
in this paper.
"""
import difflib
import gzip
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import provenance as pv  # noqa: E402

PAGES = ROOT / "cache" / "ctgov" / "pages"
OA_TERMS = {t.lower() for t in [
    "Osteoarthritis", "Spinal osteoarthritis", "Knee arthroplasty", "Hip arthroplasty",
    "Joint arthroplasty", "Arthroplasty", "Shoulder arthroplasty", "Hand osteoarthritis",
    "Primary osteoarthritis", "Secondary osteoarthritis", "Osteoarthropathy",
    "Knee osteoarthritis", "Hip osteoarthritis"]}
ARTHRO_TERMS = {"knee arthroplasty", "hip arthroplasty", "joint arthroplasty",
                "arthroplasty", "shoulder arthroplasty"}
PLACEBO_RX = re.compile(r"\b(placebos?|sham|vehicle|dummy|saline|sugar pill|"
                        r"matching tablet|matched tablet)\b", re.I)
# Non-placebo comparators are NOT placebo: an unblinded "standard of care" arm is a
# different contrast, and the pre-registered primary contrast is against placebo only.
CONTROL_RX = re.compile(r"^(control|control group|standard of care|usual care|"
                        r"best supportive care|no treatment|no intervention|observation|"
                        r"waitlist|wait-list|soc)\b", re.I)
# Some trials add a summary column to the AE table; it is not an arm.
TOTAL_RX = re.compile(r"^(total|all participants|all subjects|all patients|overall|"
                      r"all randomi[sz]ed)\b", re.I)
SALT_RX = re.compile(r"\b(anhydrous|hydrate|monohydrate|dihydrate|trihydrate|hemihydrate|"
                     r"sesquihydrate|hydrochloride|hcl|sodium|disodium|trisodium|potassium|"
                     r"calcium|magnesium|mesylate|maleate|fumarate|citrate|tartrate|"
                     r"sulfate|sulphate|phosphate|acetate|besylate|succinate|bromide|"
                     r"chloride|salt|lyophilized|free base|base)\b", re.I)
DOSE_RX = re.compile(r"\b\d+(\.\d+)?\s*(mg|mcg|µg|ug|g|ml|iu|units?|%|mg/kg|mg/m2|"
                     r"mg/day|mg/d|µg/kg|mcg/kg|mmol|meq)\b", re.I)
FORM_RX = re.compile(r"\b(tablets?|capsules?|injections?|infusion|oral|iv|i\.v\.|sc|s\.c\.|"
                     r"subcutaneous|intravenous|topical|cream|gel|ointment|solution|"
                     r"suspension|extended[- ]release|xr|er|sr|cr|film[- ]coated|once|twice|"
                     r"daily|weekly|dose|doses|low|high|fixed|combination|regimen|"
                     r"prefilled|pen|syringe|drug|biological)\b", re.I)


def norm(s):
    s = (s or "").lower()
    s = re.sub(r"^(drug|biological|other)\s*:\s*", "", s)
    s = DOSE_RX.sub(" ", s)
    s = FORM_RX.sub(" ", s)
    s = re.sub(r"[\(\)\[\],;:®™]", " ", s)
    s = SALT_RX.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip(" -+/")


def code_key(s):
    """Research codes differ only in punctuation: 'PF-05231023' == 'PF 05231023'."""
    return re.sub(r"[\s\-_]", "", (s or "").lower())


def build_index():
    """Name -> ChEMBL id.

    The plain INN is NOT in molecule_synonyms for many drugs (saxagliptin, docetaxel,
    cyclophosphamide, zoledronic acid...). It lives in the record's separate `synonyms`
    field WITH a nomenclature tag -- "Saxagliptin (INN, MI, USAN)" -- and /drug returns
    no pref_name at all. Indexing only molecule_synonyms left half of all active arms
    unmapped, including a DPP-4 inhibitor the positive controls depend on.
    """
    with pv.connect(read_only=True) as con:
        atc = {r[0]: list(r[1] or []) for r in con.execute(
            "SELECT chembl_id, atc4 FROM ref_chembl_drug").fetchall()}
    idx = defaultdict(set)
    for fpath in sorted((ROOT / "cache" / "chembl").glob("drug_*.json")):
        for d in json.loads(fpath.read_text())["drugs"]:
            cid = d["molecule_chembl_id"]
            names = [x.get("molecule_synonym") for x in (d.get("molecule_synonyms") or [])]
            names += [re.sub(r"\s*\([^)]*\)\s*$", "", x or "") for x in (d.get("synonyms") or [])]
            for x in names:
                k = norm(x)
                if len(k) >= 3:
                    idx[k].add(cid)
            for rc in d.get("research_codes") or []:
                if rc and len(code_key(rc)) >= 4:
                    idx["#" + code_key(rc)].add(cid)
    return idx, atc


def match(name, meshes, idx):
    n = norm(name)
    if not n or PLACEBO_RX.search(n):
        return set(), "placebo_or_empty"
    if n in idx:
        return idx[n], "exact"
    if "#" + code_key(name) in idx:
        return idx["#" + code_key(name)], "research_code"
    for m in meshes:                                   # the trial's own MeSH terms
        mk = norm(m)
        if mk and mk in n and mk in idx:
            return idx[mk], "mesh"
    parts = [p.strip() for p in re.split(r"\s+(?:and|with|plus)\s+|[+/&]", n) if p.strip()]
    if len(parts) > 1:
        hit = set()
        for p in parts:
            if p in idx:
                hit |= idx[p]
        if hit:
            return hit, "combination_parts"
    # hyphen-split too, so formulation prefixes ("nab-paclitaxel") fall back to the drug
    for tok in sorted(set(re.split(r"[\s\-]", n)), key=len, reverse=True):
        if len(tok) >= 5 and tok in idx:
            return idx[tok], "token"
    return set(), "unmapped"


def sim(a, b):
    return difflib.SequenceMatcher(None, (a or "").lower(), (b or "").lower()).ratio()


def containment(label, text):
    """Fraction of an arm label's tokens present in the event-group text, and the position
    of the label's most distinctive token (numbers and long words) for tie-breaking.

    Needed because many trials give event groups opaque titles ("Group I" ... "Group V",
    2,006 corpus-wide) and put the arm identity only in the DESCRIPTION. CANTOS is one:
    title matching scored 0.18 against every arm and the trial dropped out entirely.
    """
    toks = [t for t in re.findall(r"[a-z0-9]+", (label or "").lower()) if t not in
            ("dose", "arm", "group", "the", "and", "with", "of", "mg")]
    low = (text or "").lower()
    if not toks:
        return 0.0, 10 ** 9
    hit = [t for t in toks if re.search(r"\b" + re.escape(t) + r"\b", low)]
    distinct = sorted(hit, key=lambda t: (not t.isdigit(), -len(t)))
    pos = low.find(distinct[0]) if distinct else 10 ** 9
    return len(hit) / len(toks), pos


def main() -> int:
    pages = sorted(PAGES.glob("page_*.json.gz"))
    if not pages:
        print("no pages yet"); return 1
    idx, atc = build_index()
    print(f"ChEMBL synonym index: {len(idx):,} keys; pages to parse: {len(pages)}")
    trials, egs = [], []
    term_rows = defaultdict(list)
    qc = Counter()
    unmapped = Counter()
    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        for pg in pages:
            blk = pg.stem.split("_")[1].split(".")[0]
            for st in json.loads(gzip.open(pg, "rt").read())["studies"]:
                ps, rs, ds = st.get("protocolSection", {}), st.get("resultsSection", {}), \
                    st.get("derivedSection", {})
                nct = ps.get("identificationModule", {}).get("nctId")
                ae = rs.get("adverseEventsModule", {})
                if not nct or not ae:
                    qc["no_ae_module"] += 1; continue
                meshes = [m.get("term") for m in
                          ds.get("interventionBrowseModule", {}).get("meshes", []) if m.get("term")]
                aim = ps.get("armsInterventionsModule", {})
                arms = aim.get("armGroups", [])
                ivs = aim.get("interventions", [])
                design = ps.get("designModule", {})
                groups = ae.get("eventGroups", [])
                trials.append(dict(
                    nct=nct, run_id=run.run_id,
                    title=ps.get("identificationModule", {}).get("briefTitle"),
                    phase=design.get("phases") or [],
                    masking=(design.get("designInfo", {}).get("maskingInfo", {}) or {}).get("masking"),
                    primary_purpose=design.get("designInfo", {}).get("primaryPurpose"),
                    enrollment=(design.get("enrollmentInfo") or {}).get("count"),
                    start_date=(ps.get("statusModule", {}).get("startDateStruct") or {}).get("date"),
                    conditions=ps.get("conditionsModule", {}).get("conditions") or [],
                    sponsor=(ps.get("sponsorCollaboratorsModule", {}).get("leadSponsor") or {}).get("name"),
                    ae_time_frame=(ae.get("timeFrame") or "")[:300],
                    frequency_threshold=float(ae["frequencyThreshold"]) if str(ae.get("frequencyThreshold", "")).replace(".", "", 1).isdigit() else None,
                    n_event_groups=len(groups), meshes=meshes))
                # per-group outcome accumulators
                acc = {g["id"]: defaultdict(list) for g in groups}
                for kind in ("seriousEvents", "otherEvents"):
                    for ev in ae.get(kind, []) or []:
                        term = (ev.get("term") or "").strip()
                        tl = term.lower()
                        for s in ev.get("stats", []) or []:
                            gid = s.get("groupId")
                            na, nr = s.get("numAffected"), s.get("numAtRisk")
                            term_rows["nct"].append(nct); term_rows["eg_id"].append(gid)
                            term_rows["kind"].append("serious" if kind == "seriousEvents" else "other")
                            term_rows["term"].append(term)
                            term_rows["organ_system"].append(ev.get("organSystem"))
                            term_rows["num_affected"].append(na)
                            term_rows["num_at_risk"].append(nr)
                            term_rows["page_block"].append(blk)
                            if gid not in acc or na is None:
                                continue
                            k = "S" if kind == "seriousEvents" else "O"
                            if tl in OA_TERMS:
                                acc[gid][f"oa_{k}"].append(na)
                                if tl in ARTHRO_TERMS and k == "S":
                                    acc[gid]["arthro_S"].append(na)
                                if tl == "osteoarthritis" and k == "S":
                                    acc[gid]["oa_S_oa"].append(na)
                            elif tl == "arthralgia":
                                acc[gid][f"arth_{k}"].append(na)
                            elif tl == "cataract" and k == "S":
                                acc[gid]["cat_S"].append(na)
                # link event groups to arms
                for g in groups:
                    title = g.get("title") or ""
                    if TOTAL_RX.search(title.strip()):
                        qc["summary_column_excluded"] += 1
                        continue
                    desc = g.get("description") or ""
                    best, bscore = None, 0.0
                    for a in arms:
                        sc = max(sim(title, a.get("label")),
                                 sim(title, " ".join(a.get("interventionNames") or [])))
                        if sc > bscore:
                            best, bscore = a, sc
                    link_kind = "title_match"
                    if (best is None or bscore < 0.55) and desc:
                        # fall back to the DESCRIPTION: highest token containment, ties
                        # broken by which arm's distinctive token appears FIRST (so a core-
                        # phase dose is preferred over a later extension-phase dose)
                        cands = []
                        for a in arms:
                            cf, pos = containment(a.get("label"), title + " " + desc)
                            cands.append((cf, -pos, a))
                        cands.sort(key=lambda x: (x[0], x[1]), reverse=True)
                        if cands and cands[0][0] >= 0.75:
                            best, bscore, link_kind = cands[0][2], cands[0][0], "description_match"
                    if best is not None and bscore >= 0.55:
                        at = best.get("type") or ""
                        link = link_kind
                        role = ("placebo" if at in ("PLACEBO_COMPARATOR", "SHAM_COMPARATOR")
                                else "active" if at in ("EXPERIMENTAL", "ACTIVE_COMPARATOR", "OTHER")
                                else "other")
                        names = [iv.get("name") for iv in ivs
                                 if best.get("label") in (iv.get("armGroupLabels") or [])
                                 and (iv.get("type") in ("DRUG", "BIOLOGICAL"))]
                    else:
                        at, link, best = None, "no_arm_link", None
                        names = [title]
                        role = ("placebo" if PLACEBO_RX.search(title)
                                else "other" if CONTROL_RX.search(title.strip())
                                else "active")
                    ing, cids, methods = [], set(), []
                    for nm in names:
                        hit, mth = match(nm, meshes, idx)
                        if mth == "placebo_or_empty":
                            continue
                        methods.append(mth)
                        if hit:
                            ing.append(norm(nm)); cids |= hit
                        elif mth == "unmapped":
                            unmapped[norm(nm)] += 1
                    # a fallback 'placebo' title that nevertheless names a drug is active
                    if link == "no_arm_link" and role == "placebo" and cids:
                        role = "active"
                    a4 = sorted({c for cid in cids for c in atc.get(cid, [])})
                    q = acc.get(g["id"], {})
                    mx = lambda k: max(q[k]) if q.get(k) else 0
                    sm = lambda k: sum(q[k]) if q.get(k) else 0
                    egs.append(dict(
                        nct=nct, eg_id=g["id"], run_id=run.run_id, eg_title=title[:300],
                        arm_label=(best or {}).get("label"), arm_type=at, link_method=link,
                        link_score=round(bscore, 3), role=role, ingredients=sorted(set(ing)),
                        chembl_ids=sorted(cids), atc4=a4, map_methods=sorted(set(methods)),
                        n_risk_serious=g.get("seriousNumAtRisk"),
                        n_risk_other=g.get("otherNumAtRisk"),
                        oa_serious_max=mx("oa_S"), oa_serious_sum=sm("oa_S"),
                        oa_serious_osteoarthritis=mx("oa_S_oa"),
                        arthralgia_serious=mx("arth_S"), cataract_serious=mx("cat_S"),
                        arthroplasty_serious=mx("arthro_S"),
                        oa_other_max=mx("oa_O"), arthralgia_other=mx("arth_O")))
                    qc[f"link:{link}"] += 1
                    qc[f"role:{role}"] += 1
                    if role == "active":
                        qc["active_with_chembl" if cids else "active_no_chembl"] += 1
        # ---- write -------------------------------------------------------------
        stage = run.staging_dir() / "ctgov_ae"
        stage.mkdir(parents=True, exist_ok=True)
        # DuckDB writes the partitioned Parquet itself: pyarrow is absent from this
        # interpreter, and the project's store is DuckDB anyway.
        tdf = pd.DataFrame(term_rows)
        mem = duckdb.connect()
        mem.register("tdf", tdf)
        mem.execute(f"COPY tdf TO '{stage}' (FORMAT PARQUET, PARTITION_BY (page_block), "
                    f"OVERWRITE_OR_IGNORE)")
        mem.close()
        with pv.connect(read_only=True) as con:
            ins = [r[0] for r in con.execute(
                "SELECT asset_id FROM prov_input_asset WHERE asset_id IN "
                "('ctgov:randomized_with_results','chembl:drug_mechanism')").fetchall()]
        if "ctgov:randomized_with_results" not in ins:
            ins.append(run.register_input(
                "ctgov:randomized_with_results", asset_kind="trial_registry",
                local_path=PAGES, accession="ClinicalTrials.gov-API-v2",
                source_registry="ClinicalTrials.gov (U.S. NLM), public registry data",
                url="https://clinicaltrials.gov/api/v2/studies", genome_build="not_applicable",
                build_evidence="registry data, no genomic coordinates",
                licence="ClinicalTrials.gov public domain registry data",
                licence_url="https://clinicaltrials.gov/about-site/terms-conditions",
                notes="randomized interventional studies with posted results"))
        run.archive_parquet("ctgov_ae", stage, inputs=ins)
        run.archive_table("ctgov_trial", trials, inputs=ins, mode="replace_all")
        run.archive_table("ctgov_event_group", egs, inputs=ins, mode="replace_all")
    print(f"\ntrials parsed: {len(trials):,}   event groups: {len(egs):,}   "
          f"term rows: {len(term_rows['nct']):,}")
    for k in sorted(qc):
        print(f"  {k:<26} {qc[k]:>8,}")
    act = qc["active_with_chembl"] + qc["active_no_chembl"]
    print(f"  -> active groups mapped to ChEMBL: {qc['active_with_chembl']/max(act,1):.1%}")
    print("\ntop unmapped intervention names:")
    for n, c in unmapped.most_common(25):
        print(f"  {c:>5}  {n[:70]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
