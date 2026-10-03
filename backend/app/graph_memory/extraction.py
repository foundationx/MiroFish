"""LLM-based entity/fact extraction driven by MiroFish's generated ontology.

Zep Cloud extracts entities and relationships server-side. MemPalace stores
verbatim text and an explicit triple graph but does not extract, so the
MemPalace backend calls this extractor (through MiroFish's own OpenAI-format
LLM client, e.g. xAI Grok) for every ingested text episode.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ..utils.logger import get_logger

logger = get_logger("mirofish.graph_memory.extraction")

MAX_EPISODE_CHARS = 12_000
MAX_ENTITIES = 40
MAX_FACTS = 60

SYSTEM_PROMPT = """You extract a knowledge graph from text for a social-simulation engine.
Return ONLY a JSON object with this shape:
{
  "entities": [
    {"name": "...", "type": "<one of the allowed entity types>", "summary": "one or two sentences about this entity based on the text", "attributes": {"<attribute>": "<value>"}}
  ],
  "facts": [
    {"source": "<entity name>", "relation": "<one of the allowed relation names>", "target": "<entity name>", "fact": "a self-contained sentence stating the relationship", "valid_at": "ISO-8601 date if the text states when it became true, else null", "invalid_at": "ISO-8601 date if the text states when it stopped being true, else null"}
  ]
}
Rules:
- Use the allowed entity types exactly; if nothing fits use "Entity".
- Use the allowed relation names exactly (UPPER_SNAKE_CASE); if nothing fits use "RELATED_TO".
- Every fact's source and target must also appear in "entities".
- Only include attributes listed for that entity type.
- Do not invent facts that are not supported by the text. Prefer specific named people, organisations, products and groups.
- Every named person is a separate entity. Never fold a person into an organisation's attributes (e.g. a founder or CEO gets their own entity plus a relation to the organisation).
- Named products, events, topics or places that are the object of a relation are entities too; give them type "Entity" if no allowed type fits.
- Extract every relationship the text states or clearly implies (employment, founding, criticism, partnership, investment, reviews, announcements). Most entities should take part in at least one fact. Use "RELATED_TO" rather than dropping a relationship.
- Keep entity names short and canonical, and use the same full name every time (e.g. "Acme Corp", not "the company Acme Corp" or "Acme").
"""


def describe_ontology(ontology: Dict[str, Any] | None) -> str:
    """Render the ontology as compact text for the extraction prompt."""

    ontology = ontology or {}
    lines: List[str] = ["Allowed entity types:"]
    entity_types = ontology.get("entity_types") or []
    if not entity_types:
        lines.append("- Entity: any notable person, organisation, product, place or group")
    for entity in entity_types:
        attrs = ", ".join(a.get("name", "") for a in entity.get("attributes") or [] if a.get("name"))
        desc = (entity.get("description") or "").strip().replace("\n", " ")
        lines.append(f"- {entity.get('name')}: {desc}" + (f" (attributes: {attrs})" if attrs else ""))
    lines.append("")
    lines.append("Allowed relation names:")
    edge_types = ontology.get("edge_types") or []
    if not edge_types:
        lines.append("- RELATED_TO: any relationship")
    for edge in edge_types:
        pairs = ", ".join(
            f"{st.get('source', 'Entity')}->{st.get('target', 'Entity')}"
            for st in edge.get("source_targets") or []
        )
        desc = (edge.get("description") or "").strip().replace("\n", " ")
        lines.append(f"- {edge.get('name')}: {desc}" + (f" [{pairs}]" if pairs else ""))
    return "\n".join(lines)


@dataclass
class ExtractedEntity:
    name: str
    type: str = "Entity"
    summary: str = ""
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractedFact:
    source: str
    relation: str
    target: str
    fact: str
    valid_at: Optional[str] = None
    invalid_at: Optional[str] = None


@dataclass
class Extraction:
    entities: List[ExtractedEntity] = field(default_factory=list)
    facts: List[ExtractedFact] = field(default_factory=list)


_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}([T ][0-9:.+\-Z]*)?$")


def _clean_date(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value or value.lower() in {"null", "none", "unknown"}:
        return None
    return value if _ISO_DATE.match(value) else None


def _clean_name(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", value).strip()[:200]


def normalize_extraction(raw: Dict[str, Any], ontology: Dict[str, Any] | None) -> Extraction:
    """Validate an LLM extraction payload against the ontology."""

    ontology = ontology or {}
    entity_type_defs = {e.get("name"): e for e in ontology.get("entity_types") or [] if e.get("name")}
    edge_names = {e.get("name") for e in ontology.get("edge_types") or [] if e.get("name")}

    entities: Dict[str, ExtractedEntity] = {}
    for item in (raw.get("entities") or [])[:MAX_ENTITIES]:
        if not isinstance(item, dict):
            continue
        name = _clean_name(item.get("name"))
        if not name:
            continue
        etype = _clean_name(item.get("type")) or "Entity"
        if entity_type_defs and etype not in entity_type_defs:
            etype = "Entity"
        allowed_attrs = {
            a.get("name") for a in (entity_type_defs.get(etype, {}).get("attributes") or [])
        }
        attrs = item.get("attributes") if isinstance(item.get("attributes"), dict) else {}
        attrs = {
            str(k): (v if isinstance(v, (str, int, float, bool)) else json.dumps(v, ensure_ascii=False))
            for k, v in attrs.items()
            if k in allowed_attrs and v not in (None, "")
        }
        summary = item.get("summary") if isinstance(item.get("summary"), str) else ""
        key = name.lower()
        if key in entities:
            existing = entities[key]
            if summary and summary not in existing.summary:
                existing.summary = (existing.summary + " " + summary).strip()
            existing.attributes.update(attrs)
            if existing.type == "Entity" and etype != "Entity":
                existing.type = etype
        else:
            entities[key] = ExtractedEntity(name=name, type=etype, summary=summary.strip(), attributes=attrs)

    facts: List[ExtractedFact] = []
    for item in (raw.get("facts") or [])[:MAX_FACTS]:
        if not isinstance(item, dict):
            continue
        source = _clean_name(item.get("source"))
        target = _clean_name(item.get("target"))
        if not source or not target or source.lower() == target.lower():
            continue
        relation = re.sub(r"[^A-Za-z0-9_]+", "_", _clean_name(item.get("relation")) or "RELATED_TO").strip("_").upper()
        if not relation or (edge_names and relation not in edge_names):
            relation = "RELATED_TO"
        fact_text = item.get("fact") if isinstance(item.get("fact"), str) else ""
        fact_text = fact_text.strip() or f"{source} {relation.lower().replace('_', ' ')} {target}"
        for endpoint in (source, target):
            entities.setdefault(endpoint.lower(), ExtractedEntity(name=endpoint))
        facts.append(
            ExtractedFact(
                source=entities[source.lower()].name,
                relation=relation,
                target=entities[target.lower()].name,
                fact=fact_text[:1000],
                valid_at=_clean_date(item.get("valid_at")),
                invalid_at=_clean_date(item.get("invalid_at")),
            )
        )
    return Extraction(entities=list(entities.values()), facts=facts)


class LLMFactExtractor:
    """Calls an OpenAI-format chat model (e.g. Grok) to extract graph facts."""

    def __init__(self, llm_factory: Optional[Callable[[], Any]] = None):
        self._llm_factory = llm_factory
        self._llm = None

    @property
    def llm(self):
        if self._llm is None:
            if self._llm_factory is not None:
                self._llm = self._llm_factory()
            else:
                from ..utils.llm_client import LLMClient

                self._llm = LLMClient()
        return self._llm

    def extract(
        self,
        text: str,
        ontology: Dict[str, Any] | None,
        *,
        reference_time: Optional[str] = None,
        context: str = "",
    ) -> Extraction:
        text = (text or "").strip()
        if not text:
            return Extraction()
        user = (
            describe_ontology(ontology)
            + (f"\n\nReference time of this text: {reference_time}" if reference_time else "")
            + (
                "\n\nPRECEDING CONTEXT (only for resolving names and references in TEXT; "
                "do not extract facts that appear only here):\n" + context.strip()
                if context and context.strip() else ""
            )
            + "\n\nTEXT:\n"
            + text[:MAX_EPISODE_CHARS]
        )
        raw = self.llm.chat_json(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            temperature=0.1,
            max_tokens=4096,
            max_attempts=2,
        )
        if not isinstance(raw, dict):
            logger.warning("Extractor returned non-object JSON; ignoring episode")
            return Extraction()
        return normalize_extraction(raw, ontology)
