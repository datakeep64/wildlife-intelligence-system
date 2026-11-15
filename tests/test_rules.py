"""Scenario tests for the SPARQL alert rules (each scenario = one expected decision).

Graphs: `records=False` uses camera evidence only (no GBIF zone records), so the
test checks rule logic in isolation; `records=True` adds the reconciled GBIF data.
"""
from datetime import datetime, timedelta

import pytest

from src.reasoning.kg import rules
from src.reasoning.kg.graph import working_copy
from src.reasoning.kg.sightings import add_event

T0 = datetime(2026, 11, 10, 2, 0)  # 02:00, night


def scenario(events, records=False, threshold=0.5):
    g = working_copy(include_observations=records)
    for e in events:
        add_event(g, *e)
    rules.run(g, review_threshold=threshold)
    return g


def rules_for(g, event):
    return {a["rule"] for a in rules.active_alerts(g) if a["trigger"] == f"event_{event}"}


def reviewed(g, event):
    return any(r["sighting"] == f"event_{event}" for r in rules.review_tasks(g))


# ------------------------------------------------------------- HIGH alerts
def test_s01_person_near_verified_tiger_is_high():
    g = scenario([("t", "animal", "taman_negara", T0, "tiger", 0.93),
                  ("p", "person", "taman_negara", T0 + timedelta(hours=10))])
    assert rules.top_priority(g, "p") == "HIGH" and rules_for(g, "p") == {"A1"}
    reason = next(a["reasons"] for a in rules.active_alerts(g) if a["trigger"] == "event_p")
    assert "Malayan tiger" in reason


def test_s02_tiger_in_malaysia_is_refined_to_malayan_tiger():
    g = scenario([("t", "animal", "taman_negara", T0, "tiger", 0.93)])
    assert rules.sighting_taxa(g, "t") == ["malayan_tiger", "tiger"]


def test_s03_person_in_zone_with_cr_records_is_high():
    g = scenario([("p", "person", "sungei_buloh", T0)], records=True)  # straw-headed bulbul (CR) recorded there
    assert rules.top_priority(g, "p") == "HIGH" and "A2" in rules_for(g, "p")


def test_s04_vehicle_at_night_near_endangered_is_high():
    g = scenario([("tp", "animal", "endau_rompin", T0, "malayan_tapir", 0.9),
                  ("v", "vehicle", "endau_rompin", T0 + timedelta(hours=1))])
    assert rules.top_priority(g, "v") == "HIGH" and rules_for(g, "v") == {"A3"}


# ------------------------------------------------------------- not HIGH
def test_s05_tiger_in_another_zone_does_not_escalate():
    g = scenario([("t", "animal", "taman_negara", T0, "tiger", 0.93),
                  ("p", "person", "krau", T0 + timedelta(hours=1))])
    assert rules.top_priority(g, "p") == "LOW"


def test_s06_tiger_outside_time_window_does_not_escalate():
    g = scenario([("t", "animal", "taman_negara", T0, "tiger", 0.93),
                  ("p", "person", "taman_negara", T0 + timedelta(days=5))])
    assert rules.top_priority(g, "p") == "LOW"


def test_s07_vehicle_in_daytime_is_low():
    g = scenario([("tp", "animal", "endau_rompin", T0, "malayan_tapir", 0.9),
                  ("v", "vehicle", "endau_rompin", T0 + timedelta(hours=12))])  # 14:00
    assert rules.top_priority(g, "v") == "LOW"


def test_s08_person_near_endangered_is_medium():
    g = scenario([("tp", "animal", "endau_rompin", T0, "malayan_tapir", 0.9),
                  ("p", "person", "endau_rompin", T0 + timedelta(hours=2))])
    assert rules.top_priority(g, "p") == "MEDIUM" and rules_for(g, "p") == {"A4"}


def test_s09_person_near_least_concern_is_low():
    g = scenario([("b", "animal", "endau_rompin", T0, "wild_boar", 0.95),
                  ("p", "person", "endau_rompin", T0 + timedelta(hours=2))])
    assert rules.top_priority(g, "p") == "LOW"


def test_s10_verified_cr_sighting_alone_notifies_medium():
    g = scenario([("pg", "animal", "central_catchment", T0, "sunda_pangolin", 0.9)])
    assert rules.top_priority(g, "pg") == "MEDIUM" and rules_for(g, "pg") == {"A5"}


# ------------------------------------------------------------- uncertainty & plausibility -> review
def test_s11_low_confidence_threatened_goes_to_review_not_alert():
    g = scenario([("t", "animal", "taman_negara", T0, "tiger", 0.40),
                  ("p", "person", "taman_negara", T0 + timedelta(hours=1))])
    assert reviewed(g, "t")
    assert rules.top_priority(g, "p") == "LOW"   # unverified sightings cannot escalate
    assert rules.top_priority(g, "t") is None    # and do not notify researchers either


def test_s12_threshold_is_configurable_in_the_kg():
    g = scenario([("t", "animal", "taman_negara", T0, "tiger", 0.40)], threshold=0.3)
    assert not reviewed(g, "t")


def test_s13_tiger_in_singapore_is_implausible():
    g = scenario([("t", "animal", "bukit_timah", T0, "tiger", 0.99)])
    tasks = rules.review_tasks(g)
    assert reviewed(g, "t") and "not recorded in Singapore" in tasks[0]["reasons"]


def test_s14_habitat_mismatch_is_flagged():
    g = scenario([("l", "animal", "sungei_buloh", T0, "leopard", 0.95)])
    assert "no habitat in common" in rules.review_tasks(g)[0]["reasons"]


# ------------------------------------------------------------- engine properties
def test_s15_one_alert_per_event_and_fixpoint_is_stable():
    g = scenario([("t", "animal", "taman_negara", T0, "tiger", 0.93),
                  ("p", "person", "taman_negara", T0 + timedelta(hours=1))], records=True)
    per_event = [a for a in rules.active_alerts(g) if a["trigger"] == "event_p"]
    assert len(per_event) == 1 and per_event[0]["prio"] == "HIGH"
    assert "|" in per_event[0]["reasons"]  # equal-priority reasons folded in, not lost
    n = len(g)
    rules.run(g)
    assert len(g) == n  # re-running adds nothing: deterministic IRIs -> fixpoint
