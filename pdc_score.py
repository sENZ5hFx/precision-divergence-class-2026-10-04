#!/usr/bin/env python3
"""Precision-Divergence Class (PDC) scorer — 2026-10-04.

Research-stage classification. Not a physical discovery, not peer review,
not a patent. Scores whether a named quantity exhibits a residual that
GROWS as independent estimators get more precise.

Score formula (capped, archive-subtracted):

    40 * precision_gain * residual_growth * evidence
      * (1 - spoken) * (1 - prior_overlap)

Kill rule: if a later campaign reduces the residual as hidden-systematics
predicts, the instance leaves PDC. Coupling papers (hybrid mechanisms)
retire an instance; they do not mint a new class.

Author: Haley Ann Bird / autonomous Grok agent
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any


PRIOR_CLASSES = (
    "Dark Biosphere",
    "TAC",
    "Cadence Mismatch",
    "NCR",
    "FAC",
    "CIC",
    "JAC",
    "IOA",
    "DIC",
    "PPC",
    "UCC",
    "DCC",
    "FCC",
    "OAC",
    "UPRC",
    "CIP",
    "TPC",
    "CMC",
    "EFC",
    "IPC",
    "SBC",
    "IRC",
    "SEC",
    "RJC",
    "ANC",
    "UEDP",
    "RD Plane-Split",
    "CHL",
    "CLOSED-2026-09-30",
)

# Facets of prior classes that PDC must not re-issue.
# CIC = two clocks cannot be compared.
# FAC = a named discriminator licensed both camps.
# IRC = one published interval contains two qualitative worlds.
# ANC = known record dropped because a required axis is null.
# UPRC = extractable-dark vs gravitational-dark; unclosed redox column.
OVERLAP_OWNERS = ("CIC", "FAC", "IRC", "ANC", "UPRC", "PPC")


def _unit(name: str, value: float) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")
    return float(value)


@dataclass
class PdcCase:
    id: str
    realm: str
    quantity: str
    estimators: list[str]
    precision_gain: float  # d(precision)/dt over the last decade-ish, [0,1]
    residual_growth: float  # d(disagreement)/d(precision) sign+, [0,1]
    evidence: float  # primary-source lock, [0,1]
    spoken: float  # already named as THIS class (not as the phenomenon), [0,1]
    prior_overlap: float  # overlap with CIC/FAC/IRC/ANC/UPRC/PPC, [0,1]
    status: str  # confirmed | disputed | unsolved | competing | retired | control
    source: str
    role: str = "candidate"  # primary | co-primary | runner | control | retired
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.id or not self.id.replace("_", "").isalnum():
            raise ValueError(f"id must be snake_case alphanumeric, got {self.id!r}")
        if not self.realm.strip():
            raise ValueError("realm is required")
        if len(self.estimators) < 2:
            raise ValueError(f"{self.id}: PDC requires >=2 estimators")
        self.precision_gain = _unit("precision_gain", self.precision_gain)
        self.residual_growth = _unit("residual_growth", self.residual_growth)
        self.evidence = _unit("evidence", self.evidence)
        self.spoken = _unit("spoken", self.spoken)
        self.prior_overlap = _unit("prior_overlap", self.prior_overlap)
        allowed_status = {
            "confirmed",
            "disputed",
            "unsolved",
            "competing",
            "retired",
            "control",
        }
        if self.status not in allowed_status:
            raise ValueError(f"status must be one of {sorted(allowed_status)}")


def pdc_score(case: PdcCase) -> dict[str, Any]:
    """Score one PDC instance.

    Returns a dict with raw factors, the score, and a kill/keep ruling.
    A control is expected to land near zero. A retired instance (coupling
    found) must score low even if evidence is high.
    """
    keep = (
        40.0
        * case.precision_gain
        * case.residual_growth
        * case.evidence
        * (1.0 - case.spoken)
        * (1.0 - case.prior_overlap)
    )
    if case.status in {"retired", "control"}:
        ruling = "control_or_retired"
    elif keep < 1.0:
        ruling = "below_threshold"
    elif case.prior_overlap >= 0.60:
        ruling = "overlap_penalty"
    else:
        ruling = "keep"

    return {
        "id": case.id,
        "realm": case.realm,
        "quantity": case.quantity,
        "estimators": case.estimators,
        "score": round(keep, 4),
        "ruling": ruling,
        "role": case.role,
        "status": case.status,
        "factors": {
            "precision_gain": case.precision_gain,
            "residual_growth": case.residual_growth,
            "evidence": case.evidence,
            "spoken": case.spoken,
            "prior_overlap": case.prior_overlap,
            "unspoken": round(1.0 - case.spoken, 4),
            "novelty_vs_archive": round(1.0 - case.prior_overlap, 4),
        },
        "source": case.source,
        "notes": case.notes,
    }


def coupling_retirement(
    case_id: str,
    hybrid_paper: str | None,
    residual_after: float,
    residual_before: float,
) -> dict[str, Any]:
    """Test whether a coupling/hybrid paper retires a PDC instance.

    Inputs
    ------
    case_id : str
    hybrid_paper : citation or None
    residual_after, residual_before : [0, 1] disagreement magnitudes

    If residual_after < 0.5 * residual_before and a hybrid paper exists,
    the instance is retired from PDC (the unknown was an unowned gain
    stage, not a precision-divergence). Mid-landing without shrinkage
    does NOT retire.
    """
    if not case_id:
        raise ValueError("case_id required")
    residual_after = _unit("residual_after", residual_after)
    residual_before = _unit("residual_before", residual_before)
    if residual_before == 0.0:
        ratio = 0.0
    else:
        ratio = residual_after / residual_before
    retired = bool(hybrid_paper) and ratio < 0.5
    mode = (
        "retired_by_coupling"
        if retired
        else (
            "midland_no_close"
            if 0.5 <= ratio <= 1.0
            else ("enlarged" if ratio > 1.0 else "no_hybrid")
        )
    )
    return {
        "id": case_id,
        "hybrid_paper": hybrid_paper,
        "residual_before": residual_before,
        "residual_after": residual_after,
        "ratio": round(ratio, 4),
        "retired": retired,
        "mode": mode,
        "kill_rule_fired": retired,
    }


def third_operator_effect(
    case_id: str,
    values: list[float],
    labels: list[str],
) -> dict[str, Any]:
    """What happens when a third independent estimator is added?

    PDC predicts that a mid-landing third operator converts a two-body
    tension into a three-body tension without closing either gap.
    Nesting (third inside both error budgets) would kill PDC.
    """
    if len(values) != len(labels) or len(values) < 2:
        raise ValueError("values and labels must be same length, >= 2")
    if any(not math.isfinite(v) for v in values):
        raise ValueError("all values must be finite")
    ordered = sorted(zip(labels, values), key=lambda x: x[1])
    span = ordered[-1][1] - ordered[0][1]
    if span <= 0:
        return {
            "id": case_id,
            "n": len(values),
            "span": 0.0,
            "effect": "degenerate",
            "pdc_persists": False,
        }
    # Mid-land: interior points that do not sit on an endpoint.
    interiors = ordered[1:-1]
    if len(values) == 2:
        effect = "two_body"
        persists = True
    elif interiors:
        # Did any interior actually nest the endpoints? Only if span ~ 0.
        rel = span / max(abs(ordered[-1][1]), 1e-12)
        if rel < 0.02:
            effect = "nested_close"
            persists = False
        else:
            effect = "three_body_midland"
            persists = True
    else:
        effect = "two_body"
        persists = True
    return {
        "id": case_id,
        "n": len(values),
        "labels_low_to_high": [lab for lab, _ in ordered],
        "values_low_to_high": [round(v, 4) for _, v in ordered],
        "span": round(span, 4),
        "effect": effect,
        "pdc_persists": persists,
        "note": (
            "A mid-landing third clock is not a resolution. "
            "It is Precision-Divergence in three-body form."
            if effect == "three_body_midland"
            else ""
        ),
    }


CASES: list[PdcCase] = [
    PdcCase(
        id="hubble_h0dn",
        realm="outer space",
        quantity="H0 km/s/Mpc",
        estimators=["Planck/ACT+ΛCDM ~67.4", "CCHP/TRGB ~70", "H0DN 73.50±0.81"],
        precision_gain=0.95,
        residual_growth=0.94,
        evidence=1.00,
        spoken=0.18,  # tension is famous; PDC-as-class is not
        prior_overlap=0.32,  # CIC/FAC adjacent, not identical
        status="confirmed",
        source="H0DN Collaboration, A&A 2026, H0=73.50±0.81; Newswise 2026-10-01 recap; JWST Cepheid confirmation",
        role="primary",
        notes="H0DN replaced a single ladder with a distance network at ~1% and still sits 5–7σ from CMB+BAO. Precision spent; residual grew; single-error explanations retired.",
    ),
    PdcCase(
        id="desi_dynamical_de",
        realm="outer space",
        quantity="dark-energy equation of state w(z)",
        estimators=["ΛCDM w=-1", "DESI BAO+CMB+SN quintom", "systematics-only"],
        precision_gain=0.86,
        residual_growth=0.72,
        evidence=0.85,
        spoken=0.28,
        prior_overlap=0.38,  # PPC extra unobservable DOF
        status="competing",
        source="RAA 26:8 (2026) DESI-era review; dataset-combo dependence explicit",
        role="runner",
        notes="Preference for dynamical DE grows with BAO precision and flips with dataset combination. Partial PPC overlap.",
    ),
    PdcCase(
        id="mesopelagic_biomass",
        realm="ocean",
        quantity="global mesopelagic fish biomass (Gt)",
        estimators=["trawl ~1 Gt", "acoustic 7–10 Gt", "combined 1–20 Gt"],
        precision_gain=0.78,
        residual_growth=0.88,
        evidence=0.88,
        spoken=0.16,
        prior_overlap=0.34,  # IRC adjacent (1 vs 20 is two worlds); PDC is the time-derivative
        status="confirmed",
        source="SUMMER 2026; Proud 2019; Arostegui GCB 2026; DOSI policy brief Oct 2025",
        role="co-primary",
        notes="More methods widened the range to 20×. Gas-bladdered siphonophores remain the largest acoustic-ID unknown. Arostegui: 'unexploited frontier' is already fished.",
    ),
    PdcCase(
        id="ccz_virome_novelty",
        realm="ocean",
        quantity="fraction of viruses previously documented",
        estimators=["legacy viral databases", "CCZ sediment virome 2023–24"],
        precision_gain=0.82,
        residual_growth=0.74,
        evidence=0.92,
        spoken=0.08,
        prior_overlap=0.42,  # UPRC extraction-ahead
        status="confirmed",
        source="Hou, Zhang, Sung, Nature Communications, reported 2026-10-02/03; 99% undocumented",
        role="runner",
        notes="Sequencing precision went up; novelty rate stayed ~99%. Baseline-that-cannot-baseline for pre-mining EIA. Overlap-penalized vs UPRC.",
    ),
    PdcCase(
        id="dark_oxygen_sign",
        realm="ocean",
        quantity="benthic O2 flux sign at CCZ nodules",
        estimators=["Sweetman 2024 production", "prior landers consumption", "TMC/Frontiers artifact"],
        precision_gain=0.58,
        residual_growth=0.80,
        evidence=0.70,
        spoken=0.62,  # TAC, UPRC already own this object
        prior_overlap=0.72,
        status="disputed",
        source="Sweetman Nat. Geosci. 2024; Frontiers opinion 2025-12-19; DORI 2026 expeditions",
        role="runner",
        notes="Archive already spent TAC/UPRC on dark oxygen. Kept only as overlap-penalized runner. DORI is the test, not a confirmation.",
    ),
    PdcCase(
        id="llsvp_origin",
        realm="earth interior",
        quantity="origin of LLSVPs at CMB",
        estimators=["subducted-crust graveyard", "primordial magma-ocean residue", "Theia/core-exsolution mix"],
        precision_gain=0.68,
        residual_growth=0.64,
        evidence=0.82,
        spoken=0.18,
        prior_overlap=0.22,
        status="competing",
        source="Miyazaki et al. 2025; Lu et al. Science 2025-12-11 bridgmanite water; Nat. Comms. 2026-08-21 520-km split",
        role="co-primary",
        notes="Better tomography and mineral physics added a third origin (core-exsolution + magma ocean) without retiring the other two. Three-body.",
    ),
    PdcCase(
        id="crustal_h2_sink",
        realm="land",
        quantity="microbial sink of crustal/natural hydrogen",
        estimators=["engineering model ~0", "field pilots few%–>50%", "diffuse degassing unmeasured"],
        precision_gain=0.70,
        residual_growth=0.76,
        evidence=0.78,
        spoken=0.22,
        prior_overlap=0.48,  # UPRC hydrogen
        status="unsolved",
        source="Eos 2026-09-15 underground hydrogen; UPRC 2026-09-16 already logged the sink",
        role="runner",
        notes="Overlap with UPRC. Kept as land-realm instance of precision not closing a flux.",
    ),
    PdcCase(
        id="soil_microbial_dark",
        realm="land",
        quantity="named fraction of soil/host-associated microbes",
        estimators=["culture 0.1–1%", "metagenome OTU/ASV explosion", "function-not-taxon"],
        precision_gain=0.90,
        residual_growth=0.80,
        evidence=0.84,
        spoken=0.40,
        prior_overlap=0.40,
        status="confirmed",
        source="Biology Insights 2026-10-02; standard microbial-dark-matter bound",
        role="runner",
        notes="Sequencing precision increased the unnamed majority. Spoken as 'microbial dark matter'; unspoken as precision-divergence.",
    ),
    PdcCase(
        id="iit_gwt_adversarial",
        realm="humans",
        quantity="necessary and sufficient neural correlate of consciousness",
        estimators=["IIT", "GWT", "adversarial collaboration 2025"],
        precision_gain=0.80,
        residual_growth=0.78,
        evidence=0.86,
        spoken=0.22,
        prior_overlap=0.36,  # FAC: the collaboration was a named discriminator
        status="unsolved",
        source="IIT vs GWT adversarial test, Apr 2025, no winner; SciAm Oct 2026 'what if consciousness isn't real'",
        role="co-primary",
        notes="The discriminator that was supposed to pick a theory licensed both. FAC owns the discriminator-failure; PDC owns that sharper tests enlarged the residual.",
    ),
    PdcCase(
        id="magnetoreception_hybrid",
        realm="animals",
        quantity="avian magnetic compass sensor",
        estimators=["cryptochrome radical pair", "magnetite crystals", "PNAS 2026 hybrid"],
        precision_gain=0.84,
        residual_growth=0.18,  # residual SHRANK after coupling
        evidence=0.90,
        spoken=0.15,
        prior_overlap=0.20,
        status="retired",
        source="Xu/Hore-class hybrid, PNAS 123(8) e2524093123, 19 Feb 2026. Receptor in a living bird still not shown.",
        role="retired",
        notes="Positive control. Two rivals were an unowned gain stage: magnetite as amplifier of radical pairs. Coupling retired PDC for this instance. Direct in-vivo receptor still missing — that remainder is not PDC.",
    ),
    PdcCase(
        id="henneguya_energy",
        realm="animals",
        quantity="energy metabolism of an animal without mtDNA",
        estimators=["aerobic mitochondria (animal default)", "host-derived ATP (unknown path)"],
        precision_gain=0.55,
        residual_growth=0.50,
        evidence=0.80,
        spoken=0.12,
        prior_overlap=0.18,
        status="unsolved",
        source="Henneguya salminicola, first animal without mitochondrial genome (2020+); mechanism still unclear 2026",
        role="runner",
        notes="Not a precision-time-series. Weak PDC. Kept as animal-realm unspoken: the harpoon outlived the animal; the energy path did not get a third operator.",
    ),
    PdcCase(
        id="lithium7_control",
        realm="outer space",
        quantity="primordial Li-7 abundance",
        estimators=["BBN 3–4× higher", "Spite-plateau observations"],
        precision_gain=0.40,
        residual_growth=0.12,  # stable discrepancy, not growing with precision
        evidence=0.85,
        spoken=0.75,
        prior_overlap=0.20,
        status="control",
        source="Open 40+ years; 4–5σ; not a 2025–26 precision-enlarged residual",
        role="control",
        notes="Must score near zero. A real, spoken, stable discrepancy is not PDC.",
    ),
    PdcCase(
        id="teleios_control",
        realm="outer space",
        quantity="nature of ASKAP radio sphere G305.4–2.2",
        estimators=["SNR", "planetary nebula", "unknown"],
        precision_gain=0.12,
        residual_growth=0.08,
        evidence=0.70,
        spoken=0.05,
        prior_overlap=0.05,
        status="control",
        source="Teleios / G305.4–2.2, ASKAP 2025. Too new for a precision derivative.",
        role="control",
        notes="Must score near zero. No d(residual)/d(precision) yet.",
    ),
]


def rank_cases(cases: list[PdcCase] | None = None) -> dict[str, Any]:
    cases = cases or CASES
    rows = [pdc_score(c) for c in cases]
    rows.sort(key=lambda r: (-r["score"], r["id"]))
    keepers = [r for r in rows if r["ruling"] == "keep"]
    controls = [r for r in rows if r["role"] == "control"]
    retired = [r for r in rows if r["role"] == "retired"]
    return {
        "class": "Precision-Divergence Class (PDC)",
        "session": "2026-10-04T17:40:00Z",
        "formula": "40 * precision_gain * residual_growth * evidence * (1-spoken) * (1-prior_overlap)",
        "kill_rule": "Residual shrinks under later precision, or a coupling paper halves it.",
        "archive_subtracted": list(PRIOR_CLASSES),
        "n_cases": len(rows),
        "n_keep": len(keepers),
        "ranking": rows,
        "primary": next((r for r in rows if r["role"] == "primary"), None),
        "controls_ok": all(r["score"] < 1.0 for r in controls),
        "retired_ok": all(r["status"] == "retired" and r["factors"]["residual_growth"] < 0.30 for r in retired),
        "forbidden": [
            "Do not mint 150 per-realm discoveries (CHL + CLOSED-2026-09-30).",
            "Do not claim H0DN resolved Hubble tension.",
            "Do not claim dark oxygen is confirmed (DORI is the test).",
            "Do not treat 'dark' namings as one substance.",
            "Do not re-issue CIC, FAC, IRC, ANC, UEDP, UPRC.",
            "Do not claim a patent or priority date.",
        ],
    }


def iterate_surface(ranking: dict[str, Any]) -> dict[str, Any]:
    """Second-pass functions: coupling retirement + third-operator effect."""
    retirements = [
        coupling_retirement(
            "hubble_h0dn",
            hybrid_paper=None,  # dark QCD / EDE / local void are patches, not a coupling of the two clocks
            residual_after=0.90,  # still 5–7σ
            residual_before=0.70,  # ~4–5σ a decade ago
        ),
        coupling_retirement(
            "mesopelagic_biomass",
            hybrid_paper="SUMMER combined eDNA+acoustics+nets 2026",
            residual_after=0.85,  # still 1–20 Gt
            residual_before=0.70,
        ),
        coupling_retirement(
            "magnetoreception_hybrid",
            hybrid_paper="PNAS 123(8) e2524093123 (2026-02-19)",
            residual_after=0.18,
            residual_before=0.80,
        ),
        coupling_retirement(
            "iit_gwt_adversarial",
            hybrid_paper=None,
            residual_after=0.78,
            residual_before=0.60,
        ),
        coupling_retirement(
            "llsvp_origin",
            hybrid_paper="Miyazaki 2025 core-exsolution + magma-ocean mix (partial)",
            residual_after=0.55,
            residual_before=0.70,
        ),
        coupling_retirement(
            "dark_oxygen_sign",
            hybrid_paper=None,  # DORI is XOR, not hybrid
            residual_after=0.80,
            residual_before=0.50,
        ),
    ]
    thirds = [
        third_operator_effect(
            "hubble_h0dn",
            values=[67.4, 70.0, 73.5],
            labels=["Planck+ΛCDM", "CCHP/TRGB", "H0DN"],
        ),
        third_operator_effect(
            "mesopelagic_biomass",
            values=[1.0, 8.5, 20.0],
            labels=["trawl", "acoustic mid", "upper combined"],
        ),
        third_operator_effect(
            "llsvp_origin",
            values=[0.0, 1.0, 2.0],  # categorical ranks, not physical units
            labels=["recycled crust", "core-exsolution mix", "pure primordial"],
        ),
        third_operator_effect(
            "lithium7_control",
            values=[1.0, 3.5],
            labels=["observed Spite", "BBN predict"],
        ),
    ]
    return {
        "coupling_retirement": retirements,
        "third_operator": thirds,
        "surface_closed": True,
        "new_gap_from_iteration": (
            "Third-operator mid-landing does not retire PDC; it converts "
            "two-body tension into three-body tension. Hubble (67/70/73) is "
            "the type specimen. Only magnetoreception retired, via coupling "
            "not via a third clock."
        ),
        "ranking_primary_score": ranking["primary"]["score"] if ranking.get("primary") else None,
    }


def main() -> None:
    ranking = rank_cases()
    iteration = iterate_surface(ranking)
    out = {"ranking": ranking, "iteration": iteration}
    print(json.dumps(out, indent=2))
    with open("/workspace/artifacts/precision-divergence-class/pdc_result.json", "w") as f:
        json.dump(out, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
