"""Rule-based alerting: SPARQL INSERT rules, forward-chained to a fixpoint, in strata.

Why strata? Some rules use negation ("a sighting with NO issues is verified").
Negation is only safe once everything it negates has been fully derived, so rules
are grouped into strata and each stratum runs to a fixpoint before the next one
starts (stratified negation, as in Datalog).

Why a fixpoint? A rule's output can enable another rule. We re-run a stratum until
the graph stops growing. Alerts get deterministic IRIs (rule id + triggering event),
so re-firing a rule re-asserts the same triples - that is what guarantees termination.

Strata
  1 inference   T1 classifier output -> ofTaxon; T2 subspecies refinement by location;
                T3 subspecies inherit zone records; T4 materialise ThreatenedTaxon
  2 checks      C1 not recorded in country; C2 no habitat overlap; C3 low confidence on a threatened taxon
  3 triage      V1 verified = no issues; V2 one review task per sighting with issues
  4 alerts      A1-A7 (HIGH / MEDIUM / LOW)
  5 conflicts   R1 an alert is suppressed by a higher-priority alert for the same event

Usage: run(g) mutates g (use graph.working_copy()); then active_alerts(g), review_tasks(g).
"""
from dataclasses import dataclass

from rdflib import Graph, Literal, XSD

from src.reasoning.kg.graph import PREFIXES, WIS, local, query

NS = str(WIS)


@dataclass(frozen=True)
class Rule:
    id: str
    stratum: int
    description: str
    sparql: str


def _alert(rule_id: str, priority: str, trigger: str, extra_where: str, reason_expr: str, evidence: str = "") -> str:
    """Build an alert-creating INSERT. `trigger` is the SPARQL variable of the triggering event."""
    ev = f"wis:evidence {evidence} ;" if evidence else ""
    return f"""
INSERT {{ ?a a wis:Alert ; wis:hasPriority wis:{priority} ; wis:firedRule "{rule_id}" ;
            wis:triggeredBy {trigger} ; wis:concernsZone ?z ; {ev} wis:reason ?msg . }}
WHERE {{
  {extra_where}
  ?z rdfs:label ?zl .
  BIND(IRI(CONCAT("{NS}alert_{rule_id}_", STRAFTER(STR({trigger}), "#"))) AS ?a)
  BIND({reason_expr} AS ?msg)
}}"""


EN_LABEL = 'rdfs:label ?tl . FILTER(LANG(?tl) = "en")'
CAM_EVIDENCE = """?s a wis:AnimalSighting ; wis:verified true ; wis:inZone ?z ; wis:epochHours ?ts ; wis:ofTaxon ?t .
  wis:config wis:alertWindowHours ?win .
  FILTER(ABS(?te - ?ts) <= ?win)"""
# Inside a UNION branch, a FILTER only sees variables bound in that branch, so the
# event time must be re-bound there (without this, A3/A4 camera evidence silently never matched).
CAM_EVIDENCE_IN_UNION = "?e wis:epochHours ?te .\n  " + CAM_EVIDENCE

RULES = [
    # ---------------------------------------------------------------- stratum 1: inference
    Rule("T1", 1, "The classifier's prediction becomes the sighting's taxon",
         "INSERT { ?s wis:ofTaxon ?t } WHERE { ?s a wis:AnimalSighting ; wis:predictedTaxon ?t . }"),
    Rule("T2", 1, "If exactly one subspecies of the predicted species occurs in the zone's country, refine to it",
         """INSERT { ?s wis:ofTaxon ?sub ; wis:refinedFrom ?sp }
            WHERE { ?s a wis:AnimalSighting ; wis:predictedTaxon ?sp ; wis:inZone ?z . ?z wis:inCountry ?c .
                    ?sub wis:subspeciesOf ?sp ; wis:occursInCountry ?c .
                    FILTER NOT EXISTS { ?sub2 wis:subspeciesOf ?sp ; wis:occursInCountry ?c . FILTER(?sub2 != ?sub) } }"""),
    Rule("T3", 1, "Species-level zone records carry over to the only subspecies in that country",
         """INSERT { ?sub wis:observedIn ?z }
            WHERE { ?sp wis:observedIn ?z . ?z wis:inCountry ?c . ?sub wis:subspeciesOf ?sp ; wis:occursInCountry ?c .
                    FILTER NOT EXISTS { ?sub2 wis:subspeciesOf ?sp ; wis:occursInCountry ?c . FILTER(?sub2 != ?sub) } }"""),
    Rule("T4", 1, "A taxon with IUCN rank >= VU is a ThreatenedTaxon (materialises the OWL defined class)",
         "INSERT { ?t a wis:ThreatenedTaxon } WHERE { ?t wis:hasIUCNStatus ?st . ?st wis:statusRank ?r . FILTER(?r >= 2) }"),
    # ---------------------------------------------------------------- stratum 2: plausibility checks
    Rule("C1", 2, "Issue: predicted species is not recorded in the zone's country",
         f"""INSERT {{ ?s wis:issue ?msg }}
            WHERE {{ ?s a wis:AnimalSighting ; wis:predictedTaxon ?t ; wis:inZone ?z . ?z wis:inCountry ?c . ?c rdfs:label ?cl .
                     ?t {EN_LABEL}
                     FILTER NOT EXISTS {{ ?t wis:occursInCountry ?c }}
                     BIND(CONCAT(STR(?tl), " is not recorded in ", STR(?cl)) AS ?msg) }}"""),
    Rule("C2", 2, "Issue: predicted species shares no habitat with the zone",
         f"""INSERT {{ ?s wis:issue ?msg }}
            WHERE {{ ?s a wis:AnimalSighting ; wis:predictedTaxon ?t ; wis:inZone ?z . ?z rdfs:label ?zl .
                     ?t {EN_LABEL}
                     FILTER NOT EXISTS {{ ?t wis:livesIn ?h . ?z wis:hasHabitat ?h }}
                     BIND(CONCAT(STR(?tl), " has no habitat in common with ", STR(?zl)) AS ?msg) }}"""),
    Rule("C3", 2, "Issue: classifier confidence below the review threshold for a threatened taxon",
         f"""INSERT {{ ?s wis:issue ?msg }}
            WHERE {{ ?s a wis:AnimalSighting ; wis:predictedTaxon ?pt ; wis:confidence ?p .
                     wis:config wis:reviewThreshold ?tau . FILTER(?p < ?tau)
                     FILTER EXISTS {{ ?s wis:ofTaxon ?t . ?t a wis:ThreatenedTaxon }}
                     ?pt rdfs:label ?tl . FILTER(LANG(?tl) = "en")
                     BIND(CONCAT("classifier confidence ", STR(?p), " is below ", STR(?tau),
                                 " for threatened ", STR(?tl)) AS ?msg) }}"""),
    # ---------------------------------------------------------------- stratum 3: triage
    Rule("V1", 3, "A sighting with no issues is verified (negation: needs stratum 2 complete)",
         "INSERT { ?s wis:verified true } WHERE { ?s a wis:AnimalSighting . FILTER NOT EXISTS { ?s wis:issue ?i } }"),
    Rule("V2", 3, "A sighting with issues becomes a human review task",
         f"""INSERT {{ ?r a wis:ReviewTask ; wis:triggeredBy ?s ; wis:concernsZone ?z ; wis:reviewReason ?i }}
            WHERE {{ ?s wis:issue ?i ; wis:inZone ?z .
                     BIND(IRI(CONCAT("{NS}review_", STRAFTER(STR(?s), "#"))) AS ?r) }}"""),
    # ---------------------------------------------------------------- stratum 4: alerts
    Rule("A1", 4, "HIGH: person in a zone with a verified CR camera sighting within the window",
         _alert("A1", "HIGH", "?e",
                f"?e a wis:PersonDetection ; wis:inZone ?z ; wis:epochHours ?te .\n  {CAM_EVIDENCE}\n  ?t wis:hasIUCNStatus wis:CR ; {EN_LABEL}",
                'CONCAT("Person detected in ", STR(?zl), " near a verified camera sighting of ", STR(?tl), " (Critically Endangered)")',
                evidence="?s")),
    Rule("A2", 4, "HIGH: person in a zone where a CR taxon has precise GBIF records",
         _alert("A2", "HIGH", "?e",
                f"?e a wis:PersonDetection ; wis:inZone ?z .\n  ?t wis:observedIn ?z ; wis:hasIUCNStatus wis:CR ; {EN_LABEL}",
                'CONCAT("Person detected in ", STR(?zl), ", where ", STR(?tl), " (Critically Endangered) has been recorded")')),
    Rule("A3", 4, "HIGH: vehicle at night (19:00-06:00) in a zone with an EN/CR taxon (camera or records)",
         _alert("A3", "HIGH", "?e",
                f"""?e a wis:VehicleDetection ; wis:inZone ?z ; wis:hourOfDay ?h ; wis:epochHours ?te . FILTER(?h >= 19 || ?h < 6)
  {{ {CAM_EVIDENCE_IN_UNION} }} UNION {{ ?t wis:observedIn ?z }}
  ?t wis:hasIUCNStatus ?st ; {EN_LABEL} ?st wis:statusRank ?r . FILTER(?r >= 3)""",
                'CONCAT("Vehicle at night in ", STR(?zl), ", which holds ", STR(?tl))')),
    Rule("A4", 4, "MEDIUM: person in a zone with a VU/EN taxon (verified camera sighting or records)",
         _alert("A4", "MEDIUM", "?e",
                f"""?e a wis:PersonDetection ; wis:inZone ?z ; wis:epochHours ?te .
  {{ {CAM_EVIDENCE_IN_UNION} }} UNION {{ ?t wis:observedIn ?z }}
  ?t wis:hasIUCNStatus ?st ; {EN_LABEL} ?st wis:statusRank ?r . FILTER(?r >= 2 && ?r <= 3)""",
                'CONCAT("Person detected in ", STR(?zl), ", which holds threatened ", STR(?tl))')),
    Rule("A5", 4, "MEDIUM: verified camera sighting of a CR taxon (notify researchers)",
         _alert("A5", "MEDIUM", "?s",
                f"?s a wis:AnimalSighting ; wis:verified true ; wis:inZone ?z ; wis:ofTaxon ?t . ?t wis:hasIUCNStatus wis:CR ; {EN_LABEL}",
                'CONCAT("Verified camera sighting of ", STR(?tl), " (Critically Endangered) in ", STR(?zl))',
                evidence="?s")),
    Rule("A6", 4, "LOW: any person detected in a protected area",
         _alert("A6", "LOW", "?e", "?e a wis:PersonDetection ; wis:inZone ?z .",
                'CONCAT("Person detected in ", STR(?zl))')),
    Rule("A7", 4, "LOW: any vehicle detected in a protected area",
         _alert("A7", "LOW", "?e", "?e a wis:VehicleDetection ; wis:inZone ?z .",
                'CONCAT("Vehicle detected in ", STR(?zl))')),
    # ---------------------------------------------------------------- stratum 5: conflict resolution
    Rule("R1", 5, "One alert per event: suppressed by a higher-priority alert, or by an equal-priority alert from an earlier rule",
         """INSERT { ?a wis:suppressedBy ?b }
            WHERE { ?a a wis:Alert ; wis:triggeredBy ?e ; wis:hasPriority ?pa ; wis:firedRule ?fa . ?pa wis:priorityRank ?ra .
                    ?b a wis:Alert ; wis:triggeredBy ?e ; wis:hasPriority ?pb ; wis:firedRule ?fb . ?pb wis:priorityRank ?rb .
                    FILTER(?rb > ?ra || (?rb = ?ra && STR(?fb) < STR(?fa))) }"""),
]


def configure(g: Graph, review_threshold: float = 0.5, window_hours: float = 72) -> None:
    g.set((WIS.config, WIS.reviewThreshold, Literal(review_threshold, datatype=XSD.decimal)))
    g.set((WIS.config, WIS.alertWindowHours, Literal(window_hours, datatype=XSD.decimal)))


def run(g: Graph, review_threshold: float = 0.5, window_hours: float = 72, max_iter: int = 20) -> list[dict]:
    """Forward-chain all strata to fixpoint. Returns a trace: triples added per rule per pass."""
    configure(g, review_threshold, window_hours)
    trace = []
    for stratum in sorted({r.stratum for r in RULES}):
        rules = [r for r in RULES if r.stratum == stratum]
        for it in range(1, max_iter + 1):
            before = len(g)
            for r in rules:
                n0 = len(g)
                g.update(PREFIXES + r.sparql)
                if len(g) > n0:
                    trace.append({"stratum": stratum, "pass": it, "rule": r.id, "added": len(g) - n0})
            if len(g) == before:  # fixpoint for this stratum
                break
        else:
            raise RuntimeError(f"stratum {stratum} did not reach a fixpoint in {max_iter} passes")
    return trace


def active_alerts(g: Graph) -> list[dict]:
    """One row per triggering event. `reasons` include equal-priority alerts that were folded in."""
    rows = query("""
      SELECT ?a ?prio ?rank ?rule ?trigger ?zone (GROUP_CONCAT(DISTINCT ?reason; separator=" | ") AS ?reasons) WHERE {
        ?a a wis:Alert ; wis:hasPriority ?p ; wis:firedRule ?rule ; wis:triggeredBy ?trigger ; wis:concernsZone ?zone .
        ?p wis:priorityRank ?rank . BIND(STRAFTER(STR(?p), "#") AS ?prio)
        FILTER NOT EXISTS { ?a wis:suppressedBy ?other }
        { ?a wis:reason ?reason }
        UNION  # reasons of equal-priority alerts this one absorbed
        { ?x wis:suppressedBy ?a ; wis:hasPriority ?p ; wis:reason ?reason }
      } GROUP BY ?a ?prio ?rank ?rule ?trigger ?zone ORDER BY DESC(?rank) ?a""", g)
    for r in rows:
        r["alert"], r["trigger"], r["zone"] = local(r.pop("a")), local(r["trigger"]), local(r["zone"])
    return rows


def top_priority(g: Graph, event_id: str) -> str | None:
    for a in active_alerts(g):
        if a["trigger"] == f"event_{event_id}":
            return a["prio"]
    return None


def suppressed_alerts(g: Graph) -> list[dict]:
    return [{"alert": local(r["a"]), "by": local(r["b"])}
            for r in query("SELECT ?a ?b WHERE { ?a wis:suppressedBy ?b }", g)]


def review_tasks(g: Graph) -> list[dict]:
    rows = query("""
      SELECT ?r ?s (GROUP_CONCAT(?why; separator=" | ") AS ?reasons) WHERE {
        ?r a wis:ReviewTask ; wis:triggeredBy ?s ; wis:reviewReason ?why .
      } GROUP BY ?r ?s""", g)
    return [{"task": local(r["r"]), "sighting": local(r["s"]), "reasons": r["reasons"]} for r in rows]


def sighting_taxa(g: Graph, event_id: str) -> list[str]:
    return sorted(local(r["t"]) for r in query(f"SELECT ?t WHERE {{ wis:event_{event_id} wis:ofTaxon ?t }}", g))
