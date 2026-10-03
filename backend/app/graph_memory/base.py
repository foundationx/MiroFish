"""GraphMemory interface shared by every MiroFish graph/memory backend.

MiroFish was written directly against the Zep Cloud SDK. Rather than rewrite
every service, the interface below is the *Zep-shaped subset* MiroFish uses.
Any backend that implements it can be returned by
:func:`app.graph_memory.get_graph_memory_client` and the services keep working
unchanged:

* ``client.graph``: create / get / delete / set_ontology / add / search
* ``client.graph.node``: get / get_edges / with_raw_response.get_by_graph_id
* ``client.graph.edge``: with_raw_response.get_by_graph_id
* ``client.graph.episode``: get
* ``client.batch``: create / add / process / get / list / list_items

Returned objects only need the attributes MiroFish reads (``uuid_``, ``name``,
``labels``, ``summary``, ``attributes``, ``fact``, ``source_node_uuid``,
``target_node_uuid``, ``valid_at``, ``invalid_at``, ``created_at`` ...).

Errors: a missing graph/node/episode MUST raise :class:`NotFoundError`
(re-exported from ``zep_cloud`` so existing ``except`` clauses work for both
backends).
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable

from zep_cloud import NotFoundError as _ZepNotFoundError

# One error type for "graph / node / episode / batch does not exist" across
# every backend. Backends other than Zep raise it via ``not_found(...)``.
NotFoundError = _ZepNotFoundError


def not_found(message: str) -> NotFoundError:
    """Build a backend-neutral NotFoundError (HTTP-404 shaped)."""

    return NotFoundError(body={"message": message})


@runtime_checkable
class GraphNodeAPI(Protocol):
    def get(self, *, uuid_: str) -> Any: ...
    def get_edges(self, *, node_uuid: str) -> list[Any]: ...


@runtime_checkable
class GraphAPI(Protocol):
    node: Any
    edge: Any
    episode: Any

    def create(self, *, graph_id: str, name: str | None = None, description: str | None = None) -> Any: ...
    def get(self, graph_id: str) -> Any: ...
    def delete(self, *, graph_id: str) -> Any: ...
    def set_ontology(self, *, graph_ids: list[str], entities: dict, edges: Optional[dict] = None) -> Any: ...
    def add(self, *, graph_id: str, type: str, data: str, **kwargs: Any) -> Any: ...
    def search(self, *, graph_id: str, query: str, limit: int = 10, scope: str = "edges", reranker: str | None = None, **kwargs: Any) -> Any: ...


@runtime_checkable
class BatchAPI(Protocol):
    def create(self, *, metadata: dict | None = None) -> Any: ...
    def add(self, *, batch_id: str, items: list[Any]) -> list[Any]: ...
    def process(self, *, batch_id: str) -> Any: ...
    def get(self, *, batch_id: str) -> Any: ...
    def list(self, *, limit: int = 100, cursor: Any = None) -> Any: ...
    def list_items(self, *, batch_id: str, limit: int = 100, cursor: Any = None) -> Any: ...


@runtime_checkable
class GraphMemory(Protocol):
    """The client object MiroFish services hold as ``self.client``."""

    graph: GraphAPI
    batch: BatchAPI
