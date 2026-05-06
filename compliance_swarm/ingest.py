"""
Regulatory PDF ingestion: PyMuPDF → structured Documents → hierarchical chunking → vector index.

Implements parent–document retrieval with LlamaIndex ``HierarchicalNodeParser``: large
parent token windows (~1000 words) and smaller leaf chunks (~200 words) for embedding.
Leaf nodes carry Chapter / Article / Section metadata; parent nodes are stored in the
docstore for retrieval-time expansion.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.schema import NodeRelationship
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.vector_stores import SimpleVectorStore

from compliance_swarm.pdf_extract import extract_pdf
from compliance_swarm.regulatory_split import split_into_documents

# ~1000 English words ≈ 1300 tokens; ~200 words ≈ 260 tokens (LlamaIndex SentenceSplitter).
_DEFAULT_PARENT_TOKENS = 1300
_DEFAULT_CHILD_TOKENS = 260
_DEFAULT_CHUNK_OVERLAP = 40

logger = logging.getLogger(__name__)


def _ensure_project_cache_envs(project_root: Path) -> None:
    cache = project_root / ".cache"
    os.environ.setdefault("LLAMA_INDEX_CACHE_DIR", str(cache / "llama_index"))
    os.environ.setdefault("HF_HOME", str(cache / "huggingface"))


def _build_embed_model(backend: str, model_name: str | None):
    if backend == "huggingface":
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        name = model_name or "sentence-transformers/all-MiniLM-L6-v2"
        return HuggingFaceEmbedding(model_name=name)
    if backend == "openai":
        try:
            from llama_index.embeddings.openai import OpenAIEmbedding
        except ImportError as e:
            raise ImportError(
                "Install optional dependency: pip install llama-index-embeddings-openai"
            ) from e
        model = model_name or "text-embedding-3-small"
        return OpenAIEmbedding(model=model)
    raise ValueError(f"Unknown embedding backend: {backend}")


def _build_vector_store(kind: str, embed_dim: int):
    if kind == "simple":
        return SimpleVectorStore()
    if kind == "faiss":
        try:
            import faiss
            from llama_index.vector_stores.faiss import FaissVectorStore
        except ImportError as e:
            raise ImportError(
                "Install faiss and llama-index-vector-stores-faiss for --vector-backend faiss"
            ) from e
        index = faiss.IndexFlatL2(embed_dim)
        return FaissVectorStore(faiss_index=index)
    raise ValueError(f"Unknown vector backend: {kind}")


def ingest_regulatory_pdf(
    pdf_path: Path,
    *,
    persist_dir: Path,
    parent_chunk_tokens: int = _DEFAULT_PARENT_TOKENS,
    child_chunk_tokens: int = _DEFAULT_CHILD_TOKENS,
    chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP,
    embeddings: str = "huggingface",
    embedding_model: str | None = None,
    vector_backend: str = "simple",
) -> VectorStoreIndex:
    """
    Parse ``pdf_path``, split by regulatory headings, build a hierarchical node tree,
    embed leaf nodes, persist vector store + docstore + index (including parent nodes).
    """
    project_root = Path(__file__).resolve().parents[1]
    _ensure_project_cache_envs(project_root)

    full_text, page_spans = extract_pdf(pdf_path)
    if not full_text.strip():
        raise ValueError(f"No extractable text from PDF: {pdf_path}")

    documents = split_into_documents(
        full_text,
        page_spans,
        pdf_path,
        doc_id_prefix=f"{pdf_path.stem}_",
    )
    if not documents:
        from llama_index.core import Document

        documents = [
            Document(
                text=full_text,
                metadata={"source_file": str(pdf_path)},
            )
        ]

    node_parser = HierarchicalNodeParser.from_defaults(
        chunk_sizes=[parent_chunk_tokens, child_chunk_tokens],
        chunk_overlap=chunk_overlap,
    )
    all_nodes = node_parser.get_nodes_from_documents(documents)
    leaf_nodes = get_leaf_nodes(all_nodes)
    leaf_ids = {n.node_id for n in leaf_nodes}
    parent_nodes = [n for n in all_nodes if n.node_id not in leaf_ids]

    for node in leaf_nodes:
        rel = node.relationships.get(NodeRelationship.PARENT)
        if rel:
            node.metadata["parent_node_id"] = rel.node_id
        node.metadata["chunk_level"] = "leaf"

    for node in parent_nodes:
        node.metadata["chunk_level"] = "parent"

    embed_model = _build_embed_model(embeddings, embedding_model)
    probe = embed_model.get_query_embedding("dimension_probe")
    embed_dim = len(probe)
    vector_store = _build_vector_store(vector_backend, embed_dim)

    storage_context = StorageContext.from_defaults(
        docstore=SimpleDocumentStore(),
        vector_store=vector_store,
    )

    logger.info(
        "Building index: %s structural docs, %s total nodes, %s leaf embeddings",
        len(documents),
        len(all_nodes),
        len(leaf_nodes),
    )

    index = VectorStoreIndex(
        leaf_nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )
    storage_context.docstore.add_documents(parent_nodes)

    persist_dir.mkdir(parents=True, exist_ok=True)
    storage_context.persist(persist_dir=str(persist_dir))
    logger.info("Persisted storage context to %s", persist_dir)
    return index


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ingest a regulatory PDF with parent–child chunking (LlamaIndex)."
    )
    p.add_argument("pdf", type=Path, help="Path to regulatory PDF")
    p.add_argument(
        "--persist-dir",
        type=Path,
        default=Path("storage/regulatory_index"),
        help="Directory for LlamaIndex persist() output",
    )
    p.add_argument(
        "--parent-chunk-tokens",
        type=int,
        default=_DEFAULT_PARENT_TOKENS,
        help="Parent chunk size in tokens (~1000 words ≈ 1300)",
    )
    p.add_argument(
        "--child-chunk-tokens",
        type=int,
        default=_DEFAULT_CHILD_TOKENS,
        help="Leaf chunk size in tokens (~200 words ≈ 260)",
    )
    p.add_argument(
        "--chunk-overlap",
        type=int,
        default=_DEFAULT_CHUNK_OVERLAP,
        help="Token overlap between chunks at each hierarchy level",
    )
    p.add_argument(
        "--embeddings",
        choices=("huggingface", "openai"),
        default="huggingface",
        help="Embedding provider",
    )
    p.add_argument(
        "--embedding-model",
        default=None,
        help="Override model name (HF repo or OpenAI model id)",
    )
    p.add_argument(
        "--vector-backend",
        choices=("simple", "faiss"),
        default="simple",
        help="Vector store implementation",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    ingest_regulatory_pdf(
        args.pdf.resolve(),
        persist_dir=args.persist_dir.resolve(),
        parent_chunk_tokens=args.parent_chunk_tokens,
        child_chunk_tokens=args.child_chunk_tokens,
        chunk_overlap=args.chunk_overlap,
        embeddings=args.embeddings,
        embedding_model=args.embedding_model,
        vector_backend=args.vector_backend,
    )


if __name__ == "__main__":
    main()
