"""Perception -> reasoning adapter: turn detections into wis:Sighting triples.

One camera frame can yield several events (an animal AND a person), each its own node.
The classifier's species goes into wis:predictedTaxon. The rules, not this adapter,
decide what the KG believes (wis:ofTaxon, wis:verified).
"""
from datetime import datetime, timezone

from rdflib import Graph, Literal, Namespace, RDF, XSD

WIS = Namespace("https://datakeep64.github.io/wis/ontology#")
KIND = {"animal": WIS.AnimalSighting, "person": WIS.PersonDetection, "vehicle": WIS.VehicleDetection}


def add_event(g: Graph, event_id: str, kind: str, zone: str, when: datetime,
              taxon: str | None = None, confidence: float | None = None, camera: str | None = None) -> None:
    s = WIS[f"event_{event_id}"]
    g.add((s, RDF.type, KIND[kind]))
    g.add((s, RDF.type, WIS.Sighting))
    g.add((s, WIS.inZone, WIS[zone]))
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    g.add((s, WIS.observedAt, Literal(when.isoformat(), datatype=XSD.dateTime)))
    # numeric time helpers so rules can compare times without SPARQL duration arithmetic
    g.add((s, WIS.epochHours, Literal(round(when.timestamp() / 3600, 3), datatype=XSD.decimal)))
    g.add((s, WIS.hourOfDay, Literal(when.hour, datatype=XSD.integer)))
    if camera:
        g.add((s, WIS.capturedBy, WIS[f"camera_{camera}"]))
    if taxon:
        g.add((s, WIS.predictedTaxon, WIS[taxon]))
    if confidence is not None:
        g.add((s, WIS.confidence, Literal(round(float(confidence), 4), datatype=XSD.decimal)))


def from_pipeline(g: Graph, sighting: dict, when: datetime, zone: str, frame_id: str) -> list[str]:
    """Ingest one TriagePipeline output record. Returns the event ids created."""
    ids = []
    if sighting.get("species"):
        add_event(g, f"{frame_id}_animal", "animal", zone, when, sighting["species"], sighting["species_conf"], sighting.get("camera_id"))
        ids.append(f"{frame_id}_animal")
    if sighting.get("n_persons"):
        add_event(g, f"{frame_id}_person", "person", zone, when, camera=sighting.get("camera_id"))
        ids.append(f"{frame_id}_person")
    if sighting.get("n_vehicles"):
        add_event(g, f"{frame_id}_vehicle", "vehicle", zone, when, camera=sighting.get("camera_id"))
        ids.append(f"{frame_id}_vehicle")
    return ids
