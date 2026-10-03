"""MemPalace-backed GraphMemory (a local, free replacement for Zep Cloud).

Layout (one directory per MiroFish ``graph_id`` under ``MEMPALACE_DATA_DIR``)::

    <root>/graphs/<graph_id>/
        meta.json            graph name/description + ontology (JSON)
        kg.sqlite3           MemPalace KnowledgeGraph (entities + temporal triples)
        mirofish.sqlite3     sidecar: node summaries/attributes, edge facts, episodes
        palace/              MemPalace palace (ChromaDB + local MiniLM embeddings)
                             rooms: "episodes" (verbatim text), "facts", "nodes"
    <root>/batches/<batch_id>.json

The object returned by :class:`MemPalaceGraphMemory` mimics the subset of the
Zep Cloud SDK that MiroFish uses (see ``base.py``), so services do not need to
know which backend is active.

Semantics vs. Zep Cloud:
* Extraction is done by MiroFish's own LLM (``extraction.py``), guided by the
  ontology passed to ``set_ontology``. Zep did this server-side.
* Batches are processed in a local background thread; ``batch.get`` reports
  real progress. Episodes added with ``graph.add`` are extracted synchronously
  and report ``processed=True`` immediately afterwards.
* Search = MemPalace semantic search (local embeddings, hybrid re-rank) over
  fact/node drawers, plus exact entity-name matches from the KG. There is no
  cross-encoder reranker; the ``reranker`` argument is accepted and ignored.
* Temporal fields map to MemPalace triple validity: ``valid_at`` <-
  ``valid_from``, ``invalid_at`` <- ``valid_to``. ``expired_at`` is always None.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, List, Optional

from ..utils.logger import get_logger
from .base import not_found
from .extraction import Extraction, LLMFactExtractor

logger = get_logger("mirofish.graph_memory.mempalace")

NODE_MARK = "__n__"
EDGE_MARK = "__e__"
EPISODE_MARK = "__ep__"
_SAFE_GRAPH_ID = re.compile(r"^[A-Za-z0-9_.\-]{1,128}$")


def entity_id_for(name: str) -> str:
    """Same id normalisation as ``mempalace.knowledge_graph.KnowledgeGraph``."""

    return name.lower().replace(" ", "_").replace("'", "")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _check_graph_id(graph_id: str) -> str:
    if not isinstance(graph_id, str) or not _SAFE_GRAPH_ID.match(graph_id) or "__" in graph_id:
        raise ValueError(f"Unsupported graph_id for MemPalace backend: {graph_id!r}")
    return graph_id


def _split_uuid(value: str, mark: str) -> tuple[str, str]:
    if not isinstance(value, str) or mark not in value:
        raise not_found(f"Unknown id: {value!r}")
    graph_id, local = value.split(mark, 1)
    return graph_id, local


def ontology_from_zep_models(entities: Optional[dict], edges: Optional[dict]) -> Dict[str, Any]:
    """Convert the Pydantic classes MiroFish builds for Zep into ontology JSON."""

    def attrs_of(model: Any) -> List[Dict[str, str]]:
        fields = getattr(model, "model_fields", {}) or {}
        return [
            {"name": name, "description": (getattr(info, "description", None) or "")}
            for name, info in fields.items()
        ]

    entity_types = []
    for name, model in (entities or {}).items():
        entity_types.append(
            {
                "name": name,
                "description": (getattr(model, "__doc__", None) or "").strip(),
                "attributes": attrs_of(model),
            }
        )
    edge_types = []
    for name, spec in (edges or {}).items():
        model, source_targets = (spec if isinstance(spec, (tuple, list)) else (spec, []))
        edge_types.append(
            {
                "name": name,
                "description": (getattr(model, "__doc__", None) or "").strip(),
                "attributes": attrs_of(model),
                "source_targets": [
                    {"source": getattr(st, "source", None) or "Entity", "target": getattr(st, "target", None) or "Entity"}
                    for st in source_targets or []
                ],
            }
        )
    return {"entity_types": entity_types, "edge_types": edge_types}


class GraphStore:
    """All MemPalace state for one graph_id."""

    def __init__(self, root: Path, graph_id: str, extractor: LLMFactExtractor):
        self.graph_id = graph_id
        self.dir = root / "graphs" / graph_id
        self.extractor = extractor
        self.lock = threading.RLock()
        self._kg = None
        self._collection = None
        self._db: Optional[sqlite3.Connection] = None

    # ── lifecycle ────────────────────────────────────────────────────────
    @property
    def meta_path(self) -> Path:
        return self.dir / "meta.json"

    @property
    def palace_path(self) -> str:
        return str(self.dir / "palace")

    def exists(self) -> bool:
        return self.meta_path.exists()

    def create(self, name: str | None, description: str | None) -> None:
        self.dir.mkdir(parents=True, exist_ok=False)
        (self.dir / "palace").mkdir(exist_ok=True)
        self._write_meta(
            {
                "graph_id": self.graph_id,
                "name": name or self.graph_id,
                "description": description or "",
                "created_at": _now(),
                "ontology": {"entity_types": [], "edge_types": []},
                "backend": "mempalace",
            }
        )

    def meta(self) -> Dict[str, Any]:
        if not self.exists():
            raise not_found(f"Graph not found: {self.graph_id}")
        return json.loads(self.meta_path.read_text(encoding="utf-8"))

    def _write_meta(self, meta: Dict[str, Any]) -> None:
        tmp = self.meta_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self.meta_path)

    def set_ontology(self, ontology: Dict[str, Any]) -> None:
        with self.lock:
            meta = self.meta()
            meta["ontology"] = ontology
            self._write_meta(meta)

    def close(self) -> None:
        with self.lock:
            if self._db is not None:
                self._db.close()
                self._db = None
            if self._kg is not None:
                try:
                    self._kg.close()
                except Exception:  # pragma: no cover - best effort
                    pass
                self._kg = None
            self._collection = None

    # ── storage handles ──────────────────────────────────────────────────
    @property
    def kg(self):
        if self._kg is None:
            from mempalace.knowledge_graph import KnowledgeGraph

            self._kg = KnowledgeGraph(db_path=str(self.dir / "kg.sqlite3"))
        return self._kg

    @property
    def collection(self):
        if self._collection is None:
            from mempalace.palace import get_collection

            self._collection = get_collection(self.palace_path)
        return self._collection

    @property
    def db(self) -> sqlite3.Connection:
        if self._db is None:
            conn = sqlite3.connect(str(self.dir / "mirofish.sqlite3"), check_same_thread=False, timeout=30)
            conn.row_factory = sqlite3.Row
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS nodes (
                    entity_id TEXT PRIMARY KEY, name TEXT NOT NULL, type TEXT DEFAULT 'Entity',
                    summary TEXT DEFAULT '', attributes TEXT DEFAULT '{}', created_at TEXT, seq INTEGER
                );
                CREATE TABLE IF NOT EXISTS edges (
                    triple_id TEXT PRIMARY KEY, name TEXT, fact TEXT, attributes TEXT DEFAULT '{}',
                    episodes TEXT DEFAULT '[]', created_at TEXT, seq INTEGER
                );
                CREATE TABLE IF NOT EXISTS episodes (
                    uuid TEXT PRIMARY KEY, content TEXT, source_description TEXT, metadata TEXT,
                    created_at TEXT, processed INTEGER DEFAULT 0, error TEXT
                );
                """
            )
            self._db = conn
        return self._db

    # ── writes ───────────────────────────────────────────────────────────
    def _upsert_drawers(self, room: str, ids: List[str], docs: List[str], extra: Optional[List[dict]] = None) -> None:
        if not ids:
            return
        metas = []
        for i, _ in enumerate(ids):
            meta = {"wing": self.graph_id, "room": room, "source_file": f"mirofish:{self.graph_id}", "filed_at": datetime.now().isoformat()}
            if extra:
                meta.update({k: v for k, v in extra[i].items() if isinstance(v, (str, int, float, bool))})
            metas.append(meta)
        self.collection.upsert(documents=docs, ids=ids, metadatas=metas)

    def record_episode(self, episode_uuid: str, content: str, *, created_at: Optional[str], source_description: str, metadata: Optional[dict]) -> None:
        with self.lock:
            self.db.execute(
                "INSERT OR IGNORE INTO episodes (uuid, content, source_description, metadata, created_at, processed) VALUES (?,?,?,?,?,0)",
                (episode_uuid, content, source_description, json.dumps(metadata or {}, ensure_ascii=False), created_at or _now()),
            )
            self.db.commit()

    def ingest_episode(self, episode_uuid: str) -> Extraction:
        """Extract facts from a recorded episode and write them to MemPalace."""

        with self.lock:
            row = self.db.execute("SELECT * FROM episodes WHERE uuid=?", (episode_uuid,)).fetchone()
            if row is None:
                raise not_found(f"Episode not found: {episode_uuid}")
            if row["processed"]:
                return Extraction()
            content = row["content"] or ""
            created_at = row["created_at"]
            ontology = self.meta().get("ontology") or {}
        try:
            extraction = self.extractor.extract(content, ontology, reference_time=created_at)
        except Exception as error:
            with self.lock:
                self.db.execute("UPDATE episodes SET processed=1, error=? WHERE uuid=?", (f"{type(error).__name__}: {error}"[:500], episode_uuid))
                self.db.commit()
            raise
        with self.lock:
            self._upsert_drawers("episodes", ["ep_" + episode_uuid.split(EPISODE_MARK)[-1]], [content])
            self._apply_extraction(extraction, episode_uuid)
            self.db.execute("UPDATE episodes SET processed=1, error=NULL WHERE uuid=?", (episode_uuid,))
            self.db.commit()
        return extraction

    def _apply_extraction(self, extraction: Extraction, episode_uuid: str) -> None:
        kg = self.kg
        touched_nodes: Dict[str, None] = {}
        for entity in extraction.entities:
            entity_id = entity_id_for(entity.name)
            existing = self.db.execute("SELECT * FROM nodes WHERE entity_id=?", (entity_id,)).fetchone()
            if existing:
                summary = existing["summary"] or ""
                if entity.summary and entity.summary not in summary:
                    summary = (summary + " " + entity.summary).strip()[:1500]
                attrs = json.loads(existing["attributes"] or "{}")
                attrs.update(entity.attributes)
                etype = existing["type"] if existing["type"] not in (None, "", "Entity") else entity.type
                self.db.execute(
                    "UPDATE nodes SET type=?, summary=?, attributes=? WHERE entity_id=?",
                    (etype, summary, json.dumps(attrs, ensure_ascii=False), entity_id),
                )
                name = existing["name"]
            else:
                etype, summary, attrs, name = entity.type, entity.summary, entity.attributes, entity.name
                seq = self.db.execute("SELECT COALESCE(MAX(seq),0)+1 FROM nodes").fetchone()[0]
                self.db.execute(
                    "INSERT INTO nodes (entity_id, name, type, summary, attributes, created_at, seq) VALUES (?,?,?,?,?,?,?)",
                    (entity_id, name, etype, summary, json.dumps(attrs, ensure_ascii=False), _now(), seq),
                )
            kg.add_entity(name, etype or "Entity", {"summary": summary, "attributes": attrs})
            touched_nodes[entity_id] = None

        for fact in extraction.facts:
            valid_from, valid_to = fact.valid_at, fact.invalid_at
            try:
                triple_id = kg.add_triple(
                    fact.source, fact.relation, fact.target,
                    valid_from=valid_from, valid_to=valid_to,
                    source_drawer_id="ep_" + episode_uuid.split(EPISODE_MARK)[-1], adapter_name="mirofish",
                )
            except ValueError:
                triple_id = kg.add_triple(
                    fact.source, fact.relation, fact.target,
                    source_drawer_id="ep_" + episode_uuid.split(EPISODE_MARK)[-1], adapter_name="mirofish",
                )
            existing = self.db.execute("SELECT * FROM edges WHERE triple_id=?", (triple_id,)).fetchone()
            if existing:
                episodes = json.loads(existing["episodes"] or "[]")
                if episode_uuid not in episodes:
                    episodes.append(episode_uuid)
                self.db.execute("UPDATE edges SET episodes=? WHERE triple_id=?", (json.dumps(episodes), triple_id))
            else:
                seq = self.db.execute("SELECT COALESCE(MAX(seq),0)+1 FROM edges").fetchone()[0]
                self.db.execute(
                    "INSERT INTO edges (triple_id, name, fact, attributes, episodes, created_at, seq) VALUES (?,?,?,?,?,?,?)",
                    (triple_id, fact.relation, fact.fact, "{}", json.dumps([episode_uuid]), _now(), seq),
                )
                self._upsert_drawers("facts", ["fact_" + triple_id], [fact.fact], [{"triple_id": triple_id}])
        self.db.commit()

        # Refresh node drawers so node search reflects merged summaries.
        ids, docs = [], []
        for entity_id in touched_nodes:
            row = self.db.execute("SELECT * FROM nodes WHERE entity_id=?", (entity_id,)).fetchone()
            if row:
                ids.append("node_" + entity_id)
                docs.append(f"{row['name']} ({row['type']}): {row['summary'] or ''}".strip())
        self._upsert_drawers("nodes", ids, docs)

    # ── reads ────────────────────────────────────────────────────────────
    def node_uuid(self, entity_id: str) -> str:
        return f"{self.graph_id}{NODE_MARK}{entity_id}"

    def _node_obj(self, row: sqlite3.Row) -> SimpleNamespace:
        etype = row["type"] or "Entity"
        labels = ["Entity"] if etype in ("Entity", "unknown") else ["Entity", etype]
        return SimpleNamespace(
            uuid_=self.node_uuid(row["entity_id"]),
            uuid=self.node_uuid(row["entity_id"]),
            name=row["name"],
            labels=labels,
            summary=row["summary"] or "",
            attributes=json.loads(row["attributes"] or "{}"),
            created_at=row["created_at"],
            graph_id=self.graph_id,
        )

    def list_nodes(self) -> List[SimpleNamespace]:
        with self.lock:
            rows = self.db.execute("SELECT * FROM nodes ORDER BY seq").fetchall()
        return [self._node_obj(r) for r in rows]

    def get_node(self, entity_id: str) -> SimpleNamespace:
        with self.lock:
            row = self.db.execute("SELECT * FROM nodes WHERE entity_id=?", (entity_id,)).fetchone()
        if row is None:
            raise not_found(f"Node not found: {self.node_uuid(entity_id)}")
        return self._node_obj(row)

    def _triples(self) -> List[dict]:
        rows: List[dict] = []
        after = 0
        while True:
            page = self.kg.dump_rows("triples", after_rowid=after, limit=1000)
            if not page:
                break
            rows.extend(page)
            after = page[-1]["_rowid"]
        return rows

    def list_edges(self, as_of: Optional[str] = None) -> List[SimpleNamespace]:
        with self.lock:
            meta = {r["triple_id"]: r for r in self.db.execute("SELECT * FROM edges").fetchall()}
            triples = self._triples()
        out = []
        for triple in triples:
            info = meta.get(triple["id"])
            if info is None:
                continue  # triples written outside MiroFish are ignored
            if as_of and not _valid_as_of(triple.get("valid_from"), triple.get("valid_to"), as_of):
                continue
            out.append(
                SimpleNamespace(
                    uuid_=f"{self.graph_id}{EDGE_MARK}{triple['id']}",
                    uuid=f"{self.graph_id}{EDGE_MARK}{triple['id']}",
                    name=info["name"] or triple["predicate"].upper(),
                    fact=info["fact"] or "",
                    fact_type=info["name"] or triple["predicate"].upper(),
                    source_node_uuid=self.node_uuid(triple["subject"]),
                    target_node_uuid=self.node_uuid(triple["object"]),
                    attributes=json.loads(info["attributes"] or "{}"),
                    created_at=info["created_at"],
                    valid_at=triple.get("valid_from"),
                    invalid_at=triple.get("valid_to"),
                    expired_at=None,
                    episodes=json.loads(info["episodes"] or "[]"),
                    _seq=info["seq"] or 0,
                )
            )
        out.sort(key=lambda e: e._seq)
        return out

    def get_episode(self, episode_uuid: str) -> SimpleNamespace:
        with self.lock:
            row = self.db.execute("SELECT * FROM episodes WHERE uuid=?", (episode_uuid,)).fetchone()
        if row is None:
            raise not_found(f"Episode not found: {episode_uuid}")
        return SimpleNamespace(
            uuid_=episode_uuid, uuid=episode_uuid, processed=bool(row["processed"]), content=row["content"],
            created_at=row["created_at"], source_description=row["source_description"], error=row["error"],
        )

    def query_entity(self, name: str, as_of: Optional[str] = None, direction: str = "both") -> list:
        """MemPalace-native lookup (exposed for callers that want it)."""

        with self.lock:
            return self.kg.query_entity(name, as_of=as_of, direction=direction)

    def search(self, query: str, limit: int, scope: str, as_of: Optional[str] = None) -> SimpleNamespace:
        from mempalace.searcher import search_memories

        limit = max(1, int(limit))
        result = SimpleNamespace(edges=[], nodes=[], episodes=[])
        q_lower = query.lower()
        if scope in ("edges", "both", None):
            edges = self.list_edges(as_of=as_of)
            by_triple = {e.uuid_.split(EDGE_MARK, 1)[1]: e for e in edges}
            picked: Dict[str, Any] = {}
            if edges:
                with self.lock:
                    hits = search_memories(query, self.palace_path, room="facts", n_results=limit).get("results", [])
                for hit in hits:
                    drawer = hit.get("drawer_id") or ""
                    edge = by_triple.get(drawer[len("fact_"):]) if drawer.startswith("fact_") else None
                    if edge is not None:
                        picked.setdefault(edge.uuid_, edge)
                # KG boost: facts touching entities named in the query.
                named = {n.uuid_ for n in self.list_nodes() if n.name and n.name.lower() in q_lower}
                for edge in edges:
                    if len(picked) >= limit:
                        break
                    if edge.source_node_uuid in named or edge.target_node_uuid in named:
                        picked.setdefault(edge.uuid_, edge)
            result.edges = list(picked.values())[:limit]
        if scope in ("nodes", "both"):
            nodes = self.list_nodes()
            by_id = {n.uuid_.split(NODE_MARK, 1)[1]: n for n in nodes}
            picked_nodes: Dict[str, Any] = {}
            for node in nodes:
                if node.name and node.name.lower() in q_lower:
                    picked_nodes.setdefault(node.uuid_, node)
            if nodes:
                with self.lock:
                    hits = search_memories(query, self.palace_path, room="nodes", n_results=limit).get("results", [])
                for hit in hits:
                    drawer = hit.get("drawer_id") or ""
                    node = by_id.get(drawer[len("node_"):]) if drawer.startswith("node_") else None
                    if node is not None:
                        picked_nodes.setdefault(node.uuid_, node)
            result.nodes = list(picked_nodes.values())[:limit]
        return result


def _valid_as_of(valid_from: Optional[str], valid_to: Optional[str], as_of: str) -> bool:
    if valid_from and str(valid_from) > as_of:
        return False
    if valid_to and str(valid_to) <= as_of:
        return False
    return True


# ── Zep-shaped facade ───────────────────────────────────────────────────────
class _Paged:
    def __init__(self, fetch: Callable[[str], List[Any]]):
        self._fetch = fetch

    def get_by_graph_id(self, graph_id: str, *, limit: int = 100, cursor: Any = None, **_: Any) -> SimpleNamespace:
        items = self._fetch(graph_id)
        offset = int(cursor or 0)
        page = items[offset: offset + int(limit)]
        nxt = offset + len(page)
        headers = {"zep-next-cursor": str(nxt)} if nxt < len(items) and page else {}
        return SimpleNamespace(data=page, headers=headers)


class _NodeAPI:
    def __init__(self, mem: "MemPalaceGraphMemory"):
        self._mem = mem
        self.with_raw_response = _Paged(lambda gid: mem._store(gid, must_exist=True).list_nodes())

    def get(self, *, uuid_: str) -> SimpleNamespace:
        graph_id, entity_id = _split_uuid(uuid_, NODE_MARK)
        return self._mem._store(graph_id, must_exist=True).get_node(entity_id)

    def get_edges(self, *, node_uuid: str) -> List[SimpleNamespace]:
        graph_id, _ = _split_uuid(node_uuid, NODE_MARK)
        store = self._mem._store(graph_id, must_exist=True)
        return [e for e in store.list_edges() if node_uuid in (e.source_node_uuid, e.target_node_uuid)]

    def get_by_graph_id(self, graph_id: str, **kwargs: Any) -> List[SimpleNamespace]:
        return self.with_raw_response.get_by_graph_id(graph_id, **kwargs).data


class _EdgeAPI:
    def __init__(self, mem: "MemPalaceGraphMemory"):
        self.with_raw_response = _Paged(lambda gid: mem._store(gid, must_exist=True).list_edges())

    def get_by_graph_id(self, graph_id: str, **kwargs: Any) -> List[SimpleNamespace]:
        return self.with_raw_response.get_by_graph_id(graph_id, **kwargs).data


class _EpisodeAPI:
    def __init__(self, mem: "MemPalaceGraphMemory"):
        self._mem = mem

    def get(self, *, uuid_: str) -> SimpleNamespace:
        graph_id, _ = _split_uuid(uuid_, EPISODE_MARK)
        return self._mem._store(graph_id, must_exist=True).get_episode(uuid_)


class _GraphAPI:
    def __init__(self, mem: "MemPalaceGraphMemory"):
        self._mem = mem
        self.node = _NodeAPI(mem)
        self.edge = _EdgeAPI(mem)
        self.episode = _EpisodeAPI(mem)

    def create(self, *, graph_id: str, name: str | None = None, description: str | None = None, **_: Any) -> SimpleNamespace:
        store = self._mem._store(graph_id)
        if store.exists():
            raise ValueError(f"Graph already exists: {graph_id}")
        store.create(name, description)
        return SimpleNamespace(graph_id=graph_id, name=name, description=description)

    def get(self, graph_id: str) -> SimpleNamespace:
        meta = self._mem._store(graph_id, must_exist=True).meta()
        return SimpleNamespace(graph_id=graph_id, name=meta.get("name"), description=meta.get("description"), created_at=meta.get("created_at"))

    def delete(self, *, graph_id: str) -> None:
        store = self._mem._store(graph_id, must_exist=True)
        with store.lock:
            store.close()
            shutil.rmtree(store.dir, ignore_errors=True)
        self._mem._forget(graph_id)

    def set_ontology(self, *, graph_ids: List[str], entities: Optional[dict] = None, edges: Optional[dict] = None, **_: Any) -> None:
        ontology = ontology_from_zep_models(entities, edges)
        for graph_id in graph_ids:
            self._mem._store(graph_id, must_exist=True).set_ontology(ontology)

    def add(self, *, graph_id: str, type: str = "text", data: str = "", created_at: Optional[str] = None,
            source_description: str = "", metadata: Optional[dict] = None, **_: Any) -> SimpleNamespace:
        store = self._mem._store(graph_id, must_exist=True)
        episode_uuid = f"{graph_id}{EPISODE_MARK}{uuid.uuid4().hex}"
        store.record_episode(episode_uuid, data, created_at=created_at, source_description=source_description, metadata=metadata)
        try:
            store.ingest_episode(episode_uuid)
        except Exception as error:
            # The episode text is kept and marked processed so simulation
            # memory updates never stall; the failure is logged and recorded.
            logger.warning("MemPalace extraction failed for %s: %s", episode_uuid, error)
        return store.get_episode(episode_uuid)

    def search(self, *, graph_id: str, query: str, limit: int = 10, scope: str = "edges",
               reranker: Optional[str] = None, as_of: Optional[str] = None, **_: Any) -> SimpleNamespace:
        return self._mem._store(graph_id, must_exist=True).search(query, limit, scope, as_of=as_of)


class _BatchAPI:
    """Local stand-in for Zep's Batch API (persisted as JSON files)."""

    def __init__(self, mem: "MemPalaceGraphMemory"):
        self._mem = mem
        self._lock = threading.RLock()
        self._dir = mem.root / "batches"
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, batch_id: str) -> Path:
        if not re.match(r"^mpb_[0-9a-f]{32}$", batch_id or ""):
            raise not_found(f"Batch not found: {batch_id}")
        return self._dir / f"{batch_id}.json"

    def _load(self, batch_id: str) -> dict:
        path = self._path(batch_id)
        if not path.exists():
            raise not_found(f"Batch not found: {batch_id}")
        return json.loads(path.read_text(encoding="utf-8"))

    def _save(self, batch: dict) -> None:
        path = self._path(batch["batch_id"])
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(batch, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)

    @staticmethod
    def _summary(batch: dict) -> SimpleNamespace:
        items = batch["items"]
        total = len(items) or 1
        done = sum(1 for i in items if i["status"] in ("succeeded", "failed"))
        ok = sum(1 for i in items if i["status"] == "succeeded")
        failed = sum(1 for i in items if i["status"] == "failed")
        return SimpleNamespace(
            batch_id=batch["batch_id"], status=batch["status"], metadata=batch.get("metadata") or {},
            progress=SimpleNamespace(percent_complete=100.0 * done / total, succeeded_items=ok, failed_items=failed),
        )

    @staticmethod
    def _item(item: dict) -> SimpleNamespace:
        return SimpleNamespace(
            sequence_index=item["sequence_index"], status=item["status"], episode_uuid=item["episode_uuid"],
            source_uuid=item["episode_uuid"], error=item.get("error"),
        )

    def create(self, *, metadata: Optional[dict] = None, **_: Any) -> SimpleNamespace:
        batch = {"batch_id": f"mpb_{uuid.uuid4().hex}", "metadata": metadata or {}, "status": "draft", "items": [], "created_at": _now()}
        with self._lock:
            self._save(batch)
        return self._summary(batch)

    def add(self, *, batch_id: str, items: Iterable[Any], **_: Any) -> List[SimpleNamespace]:
        with self._lock:
            batch = self._load(batch_id)
            if batch["status"] != "draft":
                raise ValueError(f"Batch {batch_id} is not a draft")
            added = []
            for item in items:
                graph_id = getattr(item, "graph_id", None) or (batch.get("metadata") or {}).get("graph_id")
                self._mem._store(graph_id, must_exist=True)
                record = {
                    "sequence_index": len(batch["items"]),
                    "graph_id": graph_id,
                    "data": getattr(item, "data", None) or getattr(item, "content", None) or "",
                    "metadata": getattr(item, "metadata", None) or {},
                    "source_description": getattr(item, "source_description", None) or "",
                    "created_at": getattr(item, "created_at", None),
                    "episode_uuid": f"{graph_id}{EPISODE_MARK}{uuid.uuid4().hex}",
                    "status": "pending",
                    "error": None,
                }
                batch["items"].append(record)
                added.append(self._item(record))
            self._save(batch)
        return added

    def process(self, *, batch_id: str, **_: Any) -> SimpleNamespace:
        with self._lock:
            batch = self._load(batch_id)
            if batch["status"] != "draft":
                return self._summary(batch)
            batch["status"] = "processing"
            self._save(batch)
        thread = threading.Thread(target=self._run, args=(batch_id,), daemon=True, name=f"mempalace-{batch_id}")
        thread.start()
        if self._mem.synchronous_batches:
            thread.join()
        return self.get(batch_id=batch_id)

    def _run(self, batch_id: str) -> None:
        batch = self._load(batch_id)

        def work(record: dict) -> None:
            store = self._mem._store(record["graph_id"], must_exist=True)
            store.record_episode(
                record["episode_uuid"], record["data"], created_at=record.get("created_at"),
                source_description=record.get("source_description") or "", metadata=record.get("metadata"),
            )
            try:
                store.ingest_episode(record["episode_uuid"])
                status, error = "succeeded", None
            except Exception as exc:  # recorded per item; surfaced by GraphBuilder
                logger.warning("MemPalace batch %s item %s failed: %s", batch_id, record["sequence_index"], exc)
                status, error = "failed", f"{type(exc).__name__}: {exc}"[:500]
            with self._lock:
                current = self._load(batch_id)
                current["items"][record["sequence_index"]].update(status=status, error=error)
                self._save(current)

        with ThreadPoolExecutor(max_workers=self._mem.extract_workers) as pool:
            list(pool.map(work, batch["items"]))

        with self._lock:
            current = self._load(batch_id)
            statuses = {i["status"] for i in current["items"]}
            current["status"] = "succeeded" if statuses <= {"succeeded"} else ("failed" if statuses == {"failed"} else "partial")
            self._save(current)

    def get(self, *, batch_id: str, **_: Any) -> SimpleNamespace:
        return self._summary(self._load(batch_id))

    def list(self, *, limit: int = 100, cursor: Any = None, **_: Any) -> SimpleNamespace:
        batches = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(self._dir.glob("mpb_*.json"))]
        return SimpleNamespace(batches=[self._summary(b) for b in batches], next_cursor=None)

    def list_items(self, *, batch_id: str, limit: int = 100, cursor: Any = None, **_: Any) -> SimpleNamespace:
        return SimpleNamespace(items=[self._item(i) for i in self._load(batch_id)["items"]], next_cursor=None)


class MemPalaceGraphMemory:
    """GraphMemory implementation backed by MemPalace (one palace + KG per graph)."""

    backend_name = "mempalace"

    def __init__(
        self,
        root_dir: str | os.PathLike,
        *,
        extractor: Optional[LLMFactExtractor] = None,
        extract_workers: int = 4,
        synchronous_batches: bool = False,
    ):
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.extractor = extractor or LLMFactExtractor()
        self.extract_workers = max(1, int(extract_workers))
        self.synchronous_batches = synchronous_batches
        self._stores: Dict[str, GraphStore] = {}
        self._stores_lock = threading.Lock()
        self.graph = _GraphAPI(self)
        self.batch = _BatchAPI(self)

    def _store(self, graph_id: str, *, must_exist: bool = False) -> GraphStore:
        _check_graph_id(graph_id)
        with self._stores_lock:
            store = self._stores.get(graph_id)
            if store is None:
                store = GraphStore(self.root, graph_id, self.extractor)
                self._stores[graph_id] = store
        if must_exist and not store.exists():
            raise not_found(f"Graph not found: {graph_id}")
        return store

    def _forget(self, graph_id: str) -> None:
        with self._stores_lock:
            self._stores.pop(graph_id, None)

    def query_entity(self, graph_id: str, name: str, as_of: Optional[str] = None, direction: str = "both") -> list:
        return self._store(graph_id, must_exist=True).query_entity(name, as_of=as_of, direction=direction)

    def timeline(self, graph_id: str, entity_name: Optional[str] = None, limit: int = 100) -> list:
        return self._store(graph_id, must_exist=True).kg.timeline(entity_name, limit=limit)
