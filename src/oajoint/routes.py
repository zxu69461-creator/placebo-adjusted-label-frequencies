"""Route families (gate af_route_coding): one vocabulary for trial text and
for openFDA label routes, so a stratum means the same thing on both sides."""
from __future__ import annotations

import re

FAMILIES = ("oral", "parenteral", "inhaled", "nasal", "ophthalmic", "topical", "transdermal",
            "vaginal", "rectal", "local_injection")

# formulation-specific patterns (tier 1 of precedence)
SPECIFIC = {
    "ophthalmic": r"ophthalm|eye ?drops?|intravitreal|intra-?ocular|intracameral|subconjunctival|"
                  r"suprachoroidal|(?:each|both|study|affected|one) eyes?\b|into the eye|ocular (?:solution|suspension|insert|implant)",
    "nasal": r"intra-?nasal|nasal spray|nasal (?:solution|powder|gel|drops?)|nostril|\bnasally\b",
    "inhaled": r"inhal|nebuli[sz]|dry powder|metered[- ]dose|\bmdi\b|\bdpi\b|respimat|handihaler|diskus|"
               r"ellipta|turbuhaler|breezhaler|genuair|aerosoli[sz]|\bpuffs?\b",
    "transdermal": r"transdermal|\bpatch(?:es)?\b",
    "vaginal": r"vaginal|intravaginal|pessar|vaginal ring",
    "rectal": r"\brectal|suppositor|\benemas?\b",
    "local_injection": r"intra-?articular|epidural|intrathecal|infiltrat|nerve block|perineural|intralesional|"
                       r"intravesical|intracavernos|periodontal|subgingival|intratympanic|intrapleural|"
                       r"intradermal|intra-?tumou?ral|peri-?articular|wound instillation|local anesthe|"
                       r"intra-?uterine|intra-?prostatic|intra-?detrusor|intramyocardial|intracoronary",
    "topical": r"topical|\bcream\b|ointment|(?<!soft)(?<!soft )\bgel\b|\blotion\b|\bfoam\b|(?<!sub)(?<!sub-)(?<!per)cutaneous|"
               r"applied to (?:the )?(?:skin|scalp|lesions?|affected)|shampoo|\bvarnish\b|mouthwash|"
               r"mouth ?rinse|oral rinse|toothpaste|dentifrice",
    "parenteral": r"intravenous|\bi\.?v\.?(?=\W|$)|infusion|subcutaneous|sub-cutaneous|\bs\.?c\.?(?=\W|$)|"
                  r"intramuscular|\bi\.?m\.?(?=\W|$)|auto-?injector|pre-?filled syringe|\bbolus\b",
    "oral": r"\boral(?:ly)?\b|tablets?\b|capsules?\b|by mouth|\bp\.?o\.?(?=\W|$)|\bsyrup\b|\bsachets?\b|"
            r"chewable|lozenge|sublingual|buccal|orally disintegrating|\bgranules\b|\bdrink\b|oral solution|"
            r"oral suspension|softgel|\bpills?\b",
}
GENERIC_PARENTERAL = r"\binjection|injectable|\bsyringe|\bpen\b|\binjected\b"
_RX = {k: re.compile(v, re.I) for k, v in SPECIFIC.items()}
_GEN = re.compile(GENERIC_PARENTERAL, re.I)


def families_in(text: str) -> set[str]:
    t = text or ""
    found = {k for k, rx in _RX.items() if rx.search(t)}
    # 'oral' next to a local route is the vehicle, not the route ('oral rinse' is topical)
    if "topical" in found and re.search(r"oral rinse|mouthwash|mouth ?rinse", t, re.I):
        found.discard("oral")
    # generic injection words count only when no other non-oral route is named
    if _GEN.search(t) and not (found - {"oral", "parenteral"}):
        found.add("parenteral")
    return found


def decide(found: set[str]) -> str | None:
    if not found:
        return None
    return next(iter(found)) if len(found) == 1 else "mixed"


FDA_MAP = {
    "ORAL": "oral", "BUCCAL": "oral", "SUBLINGUAL": "oral",
    "INTRAVENOUS": "parenteral", "SUBCUTANEOUS": "parenteral", "INTRAMUSCULAR": "parenteral",
    "INTRA-ARTERIAL": "parenteral", "INTRAVENOUS DRIP": "parenteral", "INTRAVENOUS BOLUS": "parenteral",
    "RESPIRATORY (INHALATION)": "inhaled", "INHALATION": "inhaled", "NASAL": "nasal",
    "OPHTHALMIC": "ophthalmic", "INTRAVITREAL": "ophthalmic", "INTRAOCULAR": "ophthalmic",
    "INTRACAMERAL": "ophthalmic", "SUBCONJUNCTIVAL": "ophthalmic", "CONJUNCTIVAL": "ophthalmic",
    "SUPRACHOROIDAL": "ophthalmic",
    "TOPICAL": "topical", "CUTANEOUS": "topical", "DENTAL": "topical", "PERIODONTAL": "topical",
    "OROPHARYNGEAL": "topical", "AURICULAR (OTIC)": "topical", "SCALP": "topical",
    "TRANSDERMAL": "transdermal", "VAGINAL": "vaginal", "RECTAL": "rectal",
    "INTRA-ARTICULAR": "local_injection", "EPIDURAL": "local_injection", "INTRATHECAL": "local_injection",
    "INFILTRATION": "local_injection", "PERINEURAL": "local_injection", "INTRALESIONAL": "local_injection",
    "INTRAVESICAL": "local_injection", "INTRACAVERNOUS": "local_injection", "INTRADERMAL": "local_injection",
    "INTRAPLEURAL": "local_injection", "INTRAUTERINE": "local_injection", "INTRATYMPANIC": "local_injection",
    "INTRACORONARY": "local_injection", "SUBGINGIVAL": "local_injection", "INTRADETRUSOR": "local_injection",
    "INTRATUMORAL": "local_injection", "PERIARTICULAR": "local_injection", "INTRACARDIAC": "local_injection",
}


def fda_family(route: str) -> str | None:
    return FDA_MAP.get((route or "").strip().upper())
