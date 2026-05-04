"""neo4j-graphrag-based entity/relation extraction.

When ``USE_GRAPHRAG_EXTRACTOR`` is on, ``extract_via_graphrag`` runs an
``LLMEntityRelationExtractor`` per chunk and converts each result into a
LangChain ``GraphDocument`` so the rest of the pipeline
(``make_relationships``, ``post_processing``, ``common_fn.save_graphDocuments_in_neo4j``)
works unchanged.

The pipeline does NOT build a lexical graph — chunk and document nodes plus
``PART_OF`` / ``FIRST_CHUNK`` / ``NEXT_CHUNK`` / ``HAS_ENTITY`` are still owned
by ``src.make_relationships``, which writes plain ``:Chunk`` / ``:Document``.

We extract per-chunk rather than batching all chunks into a single
``LLMEntityRelationExtractor.run()`` call because, with ``create_lexical_graph=False``,
the result's ``Neo4jGraph`` does not record which entity came from which chunk;
the downstream ``merge_relationship_between_chunk_and_entites`` Cypher pairs
each chunk_id with the entities extracted from that chunk, so attribution has
to be preserved one chunk at a time. The per-chunk loop costs the same number
of LLM calls as a batched call (the extractor processes chunks in parallel
internally either way).
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Optional

from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_core.documents import Document

from src.graphrag.llm_factory import get_graphrag_llm
from src.graphrag.schema_model import (
    NodeSpec,
    Pattern,
    PropertySpec,
    RelSpec,
    SchemaSpec,
)
from src.graphrag.schema_to_graphschema import to_graph_schema


def derive_schema_spec(
    schema_spec: Optional[SchemaSpec],
    allowed_nodes: list[str],
    allowed_relationships: list[tuple[str, str, str]],
    node_properties_map: Optional[dict[str, list[str]]],
    relationship_properties_map: Optional[dict[str, list[str]]],
) -> SchemaSpec:
    """Build a SchemaSpec for the extractor. If `schema_spec` was POSTed, use it
    as-is; otherwise reconstruct from the legacy form fields.
    """
    if schema_spec is not None:
        return schema_spec

    nodes_by_label: dict[str, NodeSpec] = {
        label: NodeSpec(
            label=label,
            properties=[PropertySpec(name=p) for p in (node_properties_map or {}).get(label, [])],
        )
        for label in allowed_nodes
    }

    rels_by_label: dict[str, RelSpec] = {}
    patterns: list[Pattern] = []
    for source_label, rel_label, target_label in allowed_relationships:
        rels_by_label.setdefault(
            rel_label,
            RelSpec(
                label=rel_label,
                properties=[
                    PropertySpec(name=p)
                    for p in (relationship_properties_map or {}).get(rel_label, [])
                ],
            ),
        )
        patterns.append(
            Pattern(source_label=source_label, rel_label=rel_label, target_label=target_label)
        )

    return SchemaSpec(
        source="db",
        nodes=list(nodes_by_label.values()),
        relationships=list(rels_by_label.values()),
        patterns=patterns,
    )


async def extract_via_graphrag(
    *,
    model: str,
    combined_chunk_document_list: list[Document],
    schema_spec: SchemaSpec,
    additional_instructions: Optional[str] = None,  # noqa: ARG001 — neo4j-graphrag doesn't accept this; see module docstring
) -> tuple[list[GraphDocument], int]:
    """Run neo4j-graphrag's LLM extractor over the chunk batch.

    Returns ``(graph_documents, token_usage)`` matching the legacy contract of
    ``get_graph_document_list``: one ``GraphDocument`` per input chunk, in input
    order, with empty nodes/relationships when the extractor errors on that chunk.
    """
    from neo4j_graphrag.experimental.components.entity_relation_extractor import (
        LLMEntityRelationExtractor,
        OnError,
    )
    from neo4j_graphrag.experimental.components.types import TextChunk, TextChunks

    llm, model_name, counter = get_graphrag_llm(model)
    schema_is_empty = not (schema_spec.nodes or schema_spec.relationships or schema_spec.patterns)
    logging.info(
        "graphrag extractor: model=%s, schema=%d nodes / %d rels / %d patterns%s, chunks=%d",
        model_name,
        len(schema_spec.nodes),
        len(schema_spec.relationships),
        len(schema_spec.patterns),
        " (open extraction — no schema)" if schema_is_empty else "",
        len(combined_chunk_document_list),
    )

    extractor = LLMEntityRelationExtractor(
        llm=llm,
        on_error=OnError.IGNORE,
        create_lexical_graph=False,
    )

    # When the user didn't supply a schema we let the LLM extract freely
    # (matches the legacy LLMGraphTransformer-with-empty-allowed-nodes behavior).
    # SchemaBuilder().create_schema_model rejects empty input, so we skip it.
    run_kwargs: dict[str, Any] = {}
    if not schema_is_empty:
        run_kwargs["schema"] = to_graph_schema(schema_spec)

    graph_documents: list[GraphDocument] = []
    for i, doc in enumerate(combined_chunk_document_list):
        chunks = TextChunks(
            chunks=[
                TextChunk(
                    text=doc.page_content,
                    index=0,
                    uid=_chunk_uid(doc, i),
                )
            ]
        )
        try:
            extracted = await extractor.run(chunks=chunks, **run_kwargs)
        except Exception as exc:
            logging.error("graphrag extractor failed on chunk %d: %s", i, exc, exc_info=True)
            graph_documents.append(GraphDocument(nodes=[], relationships=[], source=doc))
            continue
        graph_documents.append(_to_graph_document(extracted, doc))

    return graph_documents, counter.total_tokens


def _chunk_uid(doc: Document, idx: int) -> str:
    chunk_ids = doc.metadata.get("combined_chunk_ids") or doc.metadata.get("chunk_id") or []
    if isinstance(chunk_ids, list) and chunk_ids:
        return str(chunk_ids[0])
    if isinstance(chunk_ids, str):
        return chunk_ids
    return f"chunk-{idx}-{uuid.uuid4().hex[:8]}"


def _to_graph_document(extracted: Any, source_doc: Document) -> GraphDocument:
    """Convert one neo4j-graphrag ``Neo4jGraph`` result into a LangChain ``GraphDocument``.

    Drops any relationship whose endpoints aren't in the same result (defensive —
    shouldn't happen for a single-chunk run, but cheap to guard).
    """
    nodes_by_id: dict[str, Node] = {}
    for n in getattr(extracted, "nodes", []):
        nodes_by_id[n.id] = Node(
            id=n.id,
            type=n.label,
            properties=dict(getattr(n, "properties", {}) or {}),
        )
    relationships: list[Relationship] = []
    for r in getattr(extracted, "relationships", []):
        src = nodes_by_id.get(r.start_node_id)
        tgt = nodes_by_id.get(r.end_node_id)
        if not src or not tgt:
            continue
        relationships.append(
            Relationship(
                source=src,
                target=tgt,
                type=r.type,
                properties=dict(getattr(r, "properties", {}) or {}),
            )
        )
    return GraphDocument(
        nodes=list(nodes_by_id.values()),
        relationships=relationships,
        source=source_doc,
    )
