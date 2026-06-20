"""
WRAITH Long-Term Memory — ChromaDB Vector Store

Stores and retrieves knowledge using semantic similarity search.
This is how WRAITH "remembers" attack techniques, WAF bypass methods,
payload patterns, and skill summaries across all scans.

ChromaDB runs fully locally — no external API calls, no data leaves
the machine. The vector database persists to disk at the configured
chroma_path (default: ./data/chroma/).

Design decisions:
    - Single collection ("wraith_knowledge") for all knowledge types.
      Metadata filtering handles type separation.
    - Default embedding: all-MiniLM-L6-v2 via ChromaDB's built-in
      sentence-transformers integration (~80MB, runs on CPU).
    - Documents are chunked by the caller — this module stores and
      retrieves raw text with metadata.
"""

from __future__ import annotations

from typing import Any

from utils.logger import get_logger

logger = get_logger("memory.longterm")

# Collection name used for all WRAITH knowledge
_COLLECTION_NAME = "wraith_knowledge"


class LongTermMemory:
    """
    ChromaDB-backed semantic vector store for persistent knowledge.

    Stores text documents with metadata and retrieves them via
    semantic similarity search. Used for:
        - Skill document summaries
        - Attack technique descriptions
        - WAF bypass methods
        - CVE knowledge from the feed ingester

    Usage:
        ltm = LongTermMemory(persist_dir="./data/chroma")
        ltm.add_document(
            doc_id="wraith-skill-0042",
            text="SQL injection via User-Agent header bypasses ModSecurity...",
            metadata={"type": "skill", "attack_class": "sqli", "confidence": "HIGH"}
        )
        results = ltm.search("WAF bypass for PHP", top_k=5)
    """

    def __init__(self, persist_dir: str = "./data/chroma") -> None:
        """
        Initialize the long-term memory store.

        Args:
            persist_dir: Directory where ChromaDB persists its data.
        """
        self._persist_dir = persist_dir
        self._client = None
        self._collection = None
        self._initialized = False

        logger.info(f"LongTermMemory configured — persist_dir={persist_dir}")

    def _ensure_initialized(self) -> None:
        """Lazy-initialize ChromaDB on first use."""
        if self._initialized:
            return

        try:
            import chromadb
            from chromadb.config import Settings

            self._client = chromadb.PersistentClient(
                path=self._persist_dir,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                ),
            )

            # Get or create the knowledge collection
            self._collection = self._client.get_or_create_collection(
                name=_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )

            self._initialized = True
            doc_count = self._collection.count()
            logger.info(
                f"ChromaDB initialized — "
                f"collection='{_COLLECTION_NAME}', "
                f"documents={doc_count}"
            )

        except ImportError:
            logger.error(
                "chromadb is not installed. "
                "Install with: pip install chromadb"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise

    # ===================================================================
    # Write Operations
    # ===================================================================

    def add_document(
        self,
        doc_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Add or update a document in the vector store.

        Args:
            doc_id: Unique identifier for the document.
            text: The text content to store and index.
            metadata: Optional metadata dict for filtering
                      (e.g., {"type": "skill", "attack_class": "sqli"}).
        """
        self._ensure_initialized()

        if not text or not text.strip():
            logger.warning(f"Skipping empty document: {doc_id}")
            return

        # ChromaDB metadata values must be str, int, float, or bool
        clean_meta = self._sanitize_metadata(metadata or {})

        try:
            self._collection.upsert(
                ids=[doc_id],
                documents=[text],
                metadatas=[clean_meta],
            )
            logger.debug(
                f"Document stored: {doc_id} "
                f"({len(text)} chars, meta={list(clean_meta.keys())})"
            )
        except Exception as e:
            logger.error(f"Failed to store document {doc_id}: {e}")
            raise

    def add_documents_batch(
        self,
        doc_ids: list[str],
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> int:
        """
        Batch-add multiple documents.

        Returns the number of documents successfully added.
        """
        self._ensure_initialized()

        if not doc_ids or not texts:
            return 0

        if len(doc_ids) != len(texts):
            raise ValueError(
                f"doc_ids ({len(doc_ids)}) and texts ({len(texts)}) "
                f"must have the same length"
            )

        clean_metas = [
            self._sanitize_metadata(m)
            for m in (metadatas or [{} for _ in doc_ids])
        ]

        # Filter out empty texts
        valid = [
            (did, txt, meta)
            for did, txt, meta in zip(doc_ids, texts, clean_metas)
            if txt and txt.strip()
        ]

        if not valid:
            return 0

        ids, docs, metas = zip(*valid)

        try:
            self._collection.upsert(
                ids=list(ids),
                documents=list(docs),
                metadatas=list(metas),
            )
            logger.info(f"Batch stored {len(ids)} documents")
            return len(ids)
        except Exception as e:
            logger.error(f"Batch store failed: {e}")
            raise

    def delete(self, doc_id: str) -> None:
        """Remove a document from the vector store."""
        self._ensure_initialized()

        try:
            self._collection.delete(ids=[doc_id])
            logger.debug(f"Document deleted: {doc_id}")
        except Exception as e:
            logger.warning(f"Failed to delete document {doc_id}: {e}")

    # ===================================================================
    # Read Operations
    # ===================================================================

    def search(
        self,
        query: str,
        top_k: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Semantic similarity search.

        Args:
            query: Natural language search query.
            top_k: Maximum number of results to return.
            where: Optional ChromaDB metadata filter
                   (e.g., {"type": "skill"} or {"attack_class": "sqli"}).

        Returns:
            List of result dicts, each containing:
                - id: Document ID
                - text: Document text
                - metadata: Document metadata
                - distance: Cosine distance (lower = more similar)
        """
        self._ensure_initialized()

        if not query or not query.strip():
            return []

        try:
            kwargs = {
                "query_texts": [query],
                "n_results": min(top_k, self._collection.count() or 1),
            }
            if where:
                kwargs["where"] = where

            results = self._collection.query(**kwargs)

            # Unpack ChromaDB's nested list format
            output = []
            if results and results.get("ids"):
                ids = results["ids"][0]
                documents = results["documents"][0] if results.get("documents") else [""] * len(ids)
                metadatas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(ids)
                distances = results["distances"][0] if results.get("distances") else [0.0] * len(ids)

                for i, doc_id in enumerate(ids):
                    output.append({
                        "id": doc_id,
                        "text": documents[i],
                        "metadata": metadatas[i],
                        "distance": distances[i],
                    })

            logger.debug(
                f"Search '{query[:50]}...' → {len(output)} results "
                f"(top distance: {output[0]['distance']:.3f})" if output else
                f"Search '{query[:50]}...' → 0 results"
            )
            return output

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def get_by_id(self, doc_id: str) -> dict[str, Any] | None:
        """Retrieve a specific document by its ID."""
        self._ensure_initialized()

        try:
            result = self._collection.get(ids=[doc_id])
            if result and result["ids"]:
                return {
                    "id": result["ids"][0],
                    "text": result["documents"][0] if result.get("documents") else "",
                    "metadata": result["metadatas"][0] if result.get("metadatas") else {},
                }
            return None
        except Exception as e:
            logger.warning(f"Get by ID failed for {doc_id}: {e}")
            return None

    def count(self) -> int:
        """Return the total number of documents in the store."""
        self._ensure_initialized()
        return self._collection.count()

    # ===================================================================
    # Maintenance
    # ===================================================================

    def reset(self) -> None:
        """Delete all documents. Use with caution."""
        self._ensure_initialized()
        self._client.delete_collection(_COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.warning("Long-term memory RESET — all documents deleted")

    # ===================================================================
    # Internal Helpers
    # ===================================================================

    @staticmethod
    def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        """
        Ensure all metadata values are ChromaDB-compatible types.
        ChromaDB only accepts: str, int, float, bool.
        """
        clean = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                clean[key] = value
            elif isinstance(value, list):
                # Convert lists to comma-separated strings
                clean[key] = ", ".join(str(v) for v in value)
            elif value is None:
                clean[key] = ""
            else:
                clean[key] = str(value)
        return clean
