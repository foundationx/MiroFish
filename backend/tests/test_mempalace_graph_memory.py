"""MemPalace graph-memory backend, exercised through the real MiroFish services.

The LLM is mocked: a fake ``chat_json`` returns a deterministic extraction so
no network/API key is needed. MemPalace itself (ChromaDB + local MiniLM
embeddings + SQLite KG) runs for real.
"""

import re

import pytest

from app import graph_memory
from app.graph_memory import NotFoundError
from app.graph_memory.extraction import LLMFactExtractor, normalize_extraction
from app.graph_memory.mempalace_backend import MemPalaceGraphMemory

ONTOLOGY = {
    "entity_types": [
        {"name": "Company", "description": "A business.", "attributes": [{"name": "industry", "type": "text", "description": "Industry"}]},
        {"name": "Person", "description": "A human.", "attributes": [{"name": "role", "type": "text", "description": "Role"}]},
        {"name": "Product", "description": "A product.", "attributes": []},
    ],
    "edge_types": [
        {"name": "LAUNCHES", "description": "Company launches product", "source_targets": [{"source": "Company", "target": "Product"}], "attributes": []},
        {"name": "WORKS_FOR", "description": "Employment", "source_targets": [{"source": "Person", "target": "Company"}], "attributes": []},
        {"name": "CRITICIZES", "description": "Criticism", "source_targets": [{"source": "Person", "target": "Product"}], "attributes": []},
    ],
}


class FakeLLM:
    """Extracts 'X launches Y', 'X works for Y', 'X criticizes Y' sentences."""

    def __init__(self):
        self.calls = 0

    def chat_json(self, messages, **_kwargs):
        self.calls += 1
        text = messages[-1]["content"].split("TEXT:\n", 1)[-1]
        entities, facts = {}, []
        patterns = [
            (r"(\w[\w ]*?) launches ([\w ]+?)\.", "Company", "Product", "LAUNCHES"),
            (r"(\w[\w ]*?) works for ([\w ]+?)\.", "Person", "Company", "WORKS_FOR"),
            (r"(\w[\w ]*?) criticizes ([\w ]+?)\.", "Person", "Product", "CRITICIZES"),
        ]
        for pattern, st, tt, rel in patterns:
            for m in re.finditer(pattern, text):
                s, o = m.group(1).strip(), m.group(2).strip()
                entities.setdefault(s, {"name": s, "type": st, "summary": f"{s} appears in the seed.", "attributes": {"role": "analyst"} if st == "Person" else {}})
                entities.setdefault(o, {"name": o, "type": tt, "summary": f"{o} is mentioned.", "attributes": {}})
                facts.append({"source": s, "relation": rel, "target": o, "fact": m.group(0), "valid_at": "2026-10-01", "invalid_at": None})
        return {"entities": list(entities.values()), "facts": facts}


@pytest.fixture()
def mempalace(tmp_path, monkeypatch):
    monkeypatch.setenv("GRAPH_MEMORY_BACKEND", "mempalace")
    monkeypatch.setenv("MEMPALACE_DATA_DIR", str(tmp_path / "mp"))
    monkeypatch.setattr("app.config.Config.ZEP_API_KEY", None)
    graph_memory.reset_graph_memory_client()
    llm = FakeLLM()
    client = MemPalaceGraphMemory(
        tmp_path / "mp",
        extractor=LLMFactExtractor(llm_factory=lambda: llm),
        synchronous_batches=True,
    )
    monkeypatch.setattr(graph_memory.factory, "_mempalace_client", client)
    yield client, llm
    graph_memory.reset_graph_memory_client()


SEED = (
    "Acme Corp launches Widget X. Dana Lee works for Acme Corp. "
    "Sam Ortiz criticizes Widget X. Priya Shah works for Beta Labs. "
    "Beta Labs launches Gadget Y."
)


def _build(builder):
    from app.services.text_processor import TextProcessor

    graph_id = builder.create_graph("Sample graph")
    builder.set_ontology(graph_id, ONTOLOGY)
    chunks = TextProcessor.split_text(SEED, 80, 0)
    submission = builder.add_text_batches(graph_id, chunks, batch_size=2)
    episodes = builder._wait_for_batch(submission)
    return graph_id, chunks, episodes


def test_backend_switch_and_config(mempalace, monkeypatch):
    from app.config import Config

    assert graph_memory.graph_memory_backend() == "mempalace"
    assert graph_memory.graph_memory_configured()
    assert not any("ZEP" in e for e in Config.validate())
    monkeypatch.setenv("GRAPH_MEMORY_BACKEND", "zep")
    assert not graph_memory.graph_memory_configured()
    monkeypatch.setenv("GRAPH_MEMORY_BACKEND", "neo4j")
    with pytest.raises(ValueError):
        graph_memory.graph_memory_backend()


def test_graph_build_lists_typed_nodes_and_temporal_edges(mempalace):
    from app.services.graph_builder import GraphBuilderService

    client, llm = mempalace
    builder = GraphBuilderService()
    assert builder.client is client
    graph_id, chunks, episodes = _build(builder)
    assert len(episodes) == len(chunks)
    assert llm.calls == len(chunks)

    data = builder.get_graph_data(graph_id)
    names = {n["name"]: n for n in data["nodes"]}
    assert {"Acme Corp", "Widget X", "Dana Lee", "Sam Ortiz", "Beta Labs", "Gadget Y", "Priya Shah"} <= set(names)
    assert names["Acme Corp"]["labels"] == ["Entity", "Company"]
    assert names["Dana Lee"]["attributes"] == {"role": "analyst"}
    facts = {e["fact"]: e for e in data["edges"]}
    edge = facts["Acme Corp launches Widget X."]
    assert edge["name"] == "LAUNCHES"
    assert edge["source_node_name"] == "Acme Corp" and edge["target_node_name"] == "Widget X"
    assert edge["valid_at"] == "2026-10-01" and edge["invalid_at"] is None
    assert edge["episodes"]

    info = builder._get_graph_info(graph_id)
    assert info.node_count == len(data["nodes"]) and info.edge_count == len(data["edges"])
    assert {"Company", "Person", "Product"} <= set(info.entity_types)

    stored = client.graph.get(graph_id)
    assert stored.name == "Sample graph"
    assert client.query_entity(graph_id, "Acme Corp")  # MemPalace-native KG lookup


def test_reader_tools_search_and_node_lookup(mempalace):
    from app.services.graph_builder import GraphBuilderService
    from app.services.zep_entity_reader import ZepEntityReader
    from app.services.zep_tools import ZepToolsService

    graph_id, _, _ = _build(GraphBuilderService())
    reader = ZepEntityReader()
    filtered = reader.filter_defined_entities(graph_id, ["Person"])
    assert {e.name for e in filtered.entities} == {"Dana Lee", "Sam Ortiz", "Priya Shah"}

    acme = next(n for n in reader.get_all_nodes(graph_id) if n["name"] == "Acme Corp")
    detail = reader.get_entity_with_context(graph_id, acme["uuid"])
    assert detail is not None and {n["name"] for n in detail.related_nodes} >= {"Widget X", "Dana Lee"}
    assert reader.get_entity_with_context(graph_id, f"{graph_id}__n__nobody") is None

    tools = ZepToolsService(llm_client=object())
    result = tools.search_graph(graph_id, "Who criticizes Widget X?", limit=5)
    assert "Sam Ortiz criticizes Widget X." in result.facts
    nodes = tools.search_graph(graph_id, "Beta Labs", limit=3, scope="nodes")
    assert any(n["name"] == "Beta Labs" for n in nodes.nodes)
    stats = tools.get_graph_statistics(graph_id)
    assert stats["total_nodes"] >= 7


def test_simulation_memory_update_episode_is_processed(mempalace):
    from app.services.graph_builder import GraphBuilderService

    client, _ = mempalace
    graph_id, _, _ = _build(GraphBuilderService())
    episode = client.graph.add(graph_id=graph_id, type="text", data="Lena Fox works for Acme Corp.",
                               created_at="2026-10-02T10:00:00Z", source_description="sim", metadata={"round": 1})
    assert client.graph.episode.get(uuid_=episode.uuid_).processed is True
    names = {n.name for n in client.graph.node.get_by_graph_id(graph_id, limit=100)}
    assert "Lena Fox" in names


def test_paging_not_found_and_delete(mempalace):
    from app.services.graph_builder import GraphBuilderService
    from app.utils.zep_paging import fetch_all_edges, fetch_all_nodes

    client, _ = mempalace
    builder = GraphBuilderService()
    graph_id, _, _ = _build(builder)
    assert len(fetch_all_nodes(client, graph_id, page_size=2)) == len(builder.get_graph_data(graph_id)["nodes"])
    assert len(fetch_all_edges(client, graph_id, page_size=1)) == 5
    with pytest.raises(NotFoundError):
        client.graph.get("mirofish_missing")
    with pytest.raises(NotFoundError):
        client.graph.episode.get(uuid_="mirofish_missing__ep__abc")
    builder.delete_graph(graph_id)
    with pytest.raises(NotFoundError):
        client.graph.get(graph_id)


def test_failed_extraction_fails_the_batch(mempalace, tmp_path):
    from app.services.graph_builder import GraphBuilderService

    client, llm = mempalace

    def boom(*_a, **_k):
        raise RuntimeError("provider 403")

    llm.chat_json = boom
    builder = GraphBuilderService()
    graph_id = builder.create_graph("x")
    submission = builder.add_text_batches(graph_id, ["Acme Corp launches Widget X."], batch_size=1)
    with pytest.raises(RuntimeError, match="ended as failed"):
        builder._wait_for_batch(submission)


def test_normalize_extraction_enforces_ontology():
    raw = {
        "entities": [{"name": "Acme", "type": "Spaceship", "summary": "s", "attributes": {"industry": "x", "bogus": 1}}],
        "facts": [{"source": "Acme", "relation": "owns stuff", "target": "Bob", "fact": "", "valid_at": "yesterday"}],
    }
    ex = normalize_extraction(raw, ONTOLOGY)
    by_name = {e.name: e for e in ex.entities}
    assert by_name["Acme"].type == "Entity" and by_name["Acme"].attributes == {}
    assert "Bob" in by_name
    assert ex.facts[0].relation == "RELATED_TO" and ex.facts[0].valid_at is None and ex.facts[0].fact
