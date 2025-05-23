# src/tools/t1_journal_context_retriever.py
import logging
import os
from typing import Any

from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


class JournalContextRetrieverArgs(BaseModel):
    """Arguments for the JournalContextRetrieverTool."""

    query_or_keywords: str = Field(
        description="The query or keywords to search for in the journal."
    )
    k_retrieval_count: int | None = Field(
        default=3, description="Number of relevant chunks to retrieve (default 3)."
    )


class JournalContextRetrieverTool(BaseTool):
    """Retrieves relevant, anonymized excerpts from the student's journal.

    This tool queries a FAISS vector store built from the apprenticeship
    journal to find specific experiences, tasks, reflections, or details
    mentioned, based on a query or keywords.
    """

    # Correction D205 (ligne 31) : Assurer la ligne vide dans la docstring de la classe.

    name: str = "journal_context_retriever"
    description: str = (
        "Retrieves relevant, anonymized excerpts from the student's "
        "apprenticeship journal based on a query or keywords. Use this to "
        "find specific experiences, tasks, reflections, or details "
        "mentioned in the journal."
    )
    args_schema: type[BaseModel] = JournalContextRetrieverArgs

    vector_store_path: str
    embedding_model_name: str

    def _run(
        self, query_or_keywords: str, k_retrieval_count: int | None = 3
    ) -> list[dict[str, Any]]:
        logger.info(f"--- Retrieving journal context for: '{query_or_keywords}' ---")

        effective_k = k_retrieval_count if k_retrieval_count is not None else 3

        try:
            embeddings = FastEmbedEmbeddings(model_name=self.embedding_model_name)
        except Exception:
            logger.error(
                f"Error initializing embedding model ({self.embedding_model_name})",
                exc_info=True,
            )
            return [
                {
                    "error": "Failed to initialize embedding model",
                }
            ]

        vector_store_index_path = os.path.join(self.vector_store_path, "index.faiss")
        if not os.path.exists(self.vector_store_path) or not os.path.exists(
            vector_store_index_path
        ):
            error_msg = f"Vector store not found at {self.vector_store_path}"
            logger.error(error_msg)
            return [{"error": error_msg}]

        try:
            vector_store = FAISS.load_local(
                self.vector_store_path,
                embeddings,
                allow_dangerous_deserialization=True,
            )
        except Exception:
            logger.error(
                f"Error loading FAISS index from {self.vector_store_path}",
                exc_info=True,
            )
            return [
                {
                    "error": "Failed to load FAISS index",
                }
            ]

        try:
            retrieved_docs_with_scores = vector_store.similarity_search_with_score(
                query_or_keywords, k=effective_k
            )
        except Exception:
            logger.error("Error during similarity search", exc_info=True)
            return [
                {
                    "error": "Error during similarity search",
                }
            ]

        output_excerpts: list[dict[str, Any]] = []
        if not retrieved_docs_with_scores:
            logger.info("No relevant results found for query.")
            return []

        for doc, score in retrieved_docs_with_scores:
            output_excerpts.append(
                {
                    "text": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score),
                }
            )
        logger.info(f"Retrieved {len(output_excerpts)} excerpts.")
        return output_excerpts

    async def _arun(
        self, query_or_keywords: str, k_retrieval_count: int | None = 3
    ) -> list[dict[str, Any]]:
        logger.debug(
            f"Async call to _arun, deferring to sync _run for: '{query_or_keywords}'"
        )
        return self._run(
            query_or_keywords=query_or_keywords,
            k_retrieval_count=k_retrieval_count,
        )
