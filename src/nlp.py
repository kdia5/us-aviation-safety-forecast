"""Negation-aware text features for aviation safety narratives.

Safety narratives are full of negated findings ("no evidence of mechanical
failure", "the pilot did not report any icing"). A naive keyword count treats
those as positive mentions. `scrub_negations` removes the negated clause before
any keyword or topic feature is computed.
"""
import re

import numpy as np

# Negation cues and the clause they govern (up to the next clause boundary).
_NEG_CUES = (
    r"no evidence of|no indication of|no indications of|no record of|no reports? of|"
    r"no anomalies|ruled out|negative for|did not|does not|was not|were not|"
    r"could not be determined|not|no|nor|without|never|denied|unremarkable"
)
_NEG_RE = re.compile(rf"\b(?:{_NEG_CUES})\b[^.;:!?]*?(?=[.;:!?]|\bbut\b|\bhowever\b|$)",
                     flags=re.IGNORECASE)

# Keyword families, grouped by the risk category they are precursors for.
KEYWORD_FAMILIES = {
    "kw_traffic_separation": [r"\btcas\b", r"traffic advisory", r"see and avoid", r"loss of separation",
                              r"near mid-?air", r"did not see the other", r"\bctaf\b", r"traffic pattern"],
    "kw_runway_surface": [r"hold short", r"taxiway", r"crossed the runway", r"wrong runway",
                          r"ground control", r"vehicle on the runway", r"runway incursion"],
    "kw_landing_control": [r"crosswind", r"gust", r"tailwind", r"wet runway", r"braking",
                           r"hydroplan", r"overran", r"veered", r"long landing", r"bounced"],
    "kw_loss_of_control": [r"\bstall", r"spatial disorientation", r"\bspin\b", r"\bicing\b",
                           r"density altitude", r"loss of control", r"airspeed decay"],
    "kw_terrain_weather": [r"\bimc\b", r"instrument meteorological", r"\bfog\b", r"low visibility",
                           r"\bterrain\b", r"mountain", r"below minimum", r"cloud"],
    "kw_powerplant_fuel": [r"loss of engine power", r"fuel exhaustion", r"fuel starvation",
                           r"engine failure", r"partial power"],
    "kw_human_factors": [r"fatigue", r"distract", r"complacen", r"inadequate preflight",
                         r"student pilot", r"workload"],
}
_FAMILY_RE = {k: re.compile("|".join(v), flags=re.IGNORECASE) for k, v in KEYWORD_FAMILIES.items()}


def scrub_negations(text: str) -> str:
    """Remove negated clauses so they are not counted as positive mentions."""
    if not isinstance(text, str):
        return ""
    return _NEG_RE.sub(" ", text)


def keyword_flags(texts):
    """Return a dict of 0/1 arrays: does each (negation-scrubbed) text mention each family?"""
    scrubbed = [scrub_negations(t) for t in texts]
    return {k: np.array([1 if r.search(t) else 0 for t in scrubbed]) for k, r in _FAMILY_RE.items()}


if __name__ == "__main__":
    # Quick self-check of the negation scrubber.
    cases = {
        "There was no evidence of a fuel exhaustion event.": "kw_powerplant_fuel",
        "The pilot did not observe any traffic advisory.": "kw_traffic_separation",
        "The engine experienced a loss of engine power during climb.": "kw_powerplant_fuel",
        "Icing was not reported, but the airplane entered a stall.": "kw_loss_of_control",
    }
    for text, fam in cases.items():
        flag = keyword_flags([text])[fam][0]
        print(f"{flag}  {fam:22s} <- {text}")
