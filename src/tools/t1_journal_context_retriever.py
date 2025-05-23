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
    """Input arguments for JournalContextRetrieverTool."""

    query_or_keywords: str = Field(
        description="La requête ou les mots-clés à rechercher dans le journal."
    )
    k_retrieval_count: int | None = Field(
        default=3, description="Nombre de chunks pertinents à récupérer (défaut 3)."
    )


class JournalContextRetrieverTool(BaseTool):
    """Tool to retrieve relevant, anonymized excerpts from journal."""

    name: str = "journal_context_retriever"
    description: str = (
        "Récupère des extraits pertinents et anonymisés du journal "
        "d'apprentissage de l'étudiant en fonction d'une requête ou de "
        "mots-clés. Utiliser pour trouver des expériences spécifiques, "
        "des tâches, des réflexions ou des détails mentionnés dans le journal."
    )
    args_schema: type[BaseModel] = JournalContextRetrieverArgs

    vector_store_path: str
    embedding_model_name: str

    def _load_vector_store(self, embeddings: FastEmbedEmbeddings) -> FAISS | None:
        """Loads the FAISS vector store from the configured path."""
        index_file = os.path.join(self.vector_store_path, "index.faiss")
        if not os.path.exists(self.vector_store_path) or not os.path.exists(index_file):
            logger.error(
                "Vector store non trouvé ou incomplet à %s", self.vector_store_path
            )
            return None
        try:
            return FAISS.load_local(
                self.vector_store_path, embeddings, allow_dangerous_deserialization=True
            )
        except FileNotFoundError as e:  # Attraper FileNotFoundError spécifiquement
            logger.error(
                "Fichier index FAISS non trouvé (%s): %s", self.vector_store_path, e
            )
            return None
        except Exception as e:  # Pour autres erreurs de chargement de FAISS
            logger.error(
                "Erreur chargement index FAISS %s: %s",
                self.vector_store_path,
                e,
                exc_info=True,
            )
            return None

    def _run(
        self, query_or_keywords: str, k_retrieval_count: int | None = None
    ) -> list[dict[str, Any]]:
        logger.info(
            "Récupération de contexte pour : '%s' avec k=%s",
            query_or_keywords,
            k_retrieval_count,
        )
        effective_k = k_retrieval_count
        if effective_k is None:
            default_k_from_schema = self.args_schema.model_fields[
                "k_retrieval_count"
            ].default
            effective_k = (
                default_k_from_schema if default_k_from_schema is not None else 3
            )

        try:
            embeddings = FastEmbedEmbeddings(model_name=self.embedding_model_name)
        except Exception as e:
            logger.error(
                "Échec init embedding model (%s): %s",
                self.embedding_model_name,
                e,
                exc_info=True,
            )
            return [{"error": "Échec init embedding model", "details": str(e)}]

        vector_store = self._load_vector_store(embeddings)
        if not vector_store:
            # Message d'erreur coupé pour respecter la longueur
            error_msg_part1 = "Vector store non trouvé ou erreur de chargement à"
            error_msg_part2 = f" {self.vector_store_path}"
            return [{"error": error_msg_part1 + error_msg_part2}]

        if vector_store.index is None or vector_store.index.ntotal == 0:
            logger.warning("L'index FAISS à %s est vide.", self.vector_store_path)
            return []

        try:
            retrieved_docs = vector_store.similarity_search_with_score(
                query_or_keywords, k=effective_k
            )
        except Exception as e:
            logger.error(
                "Erreur lors de la recherche de similarité : %s", e, exc_info=True
            )
            return [{"error": "Erreur recherche similarité", "details": str(e)}]

        output_excerpts: list[dict[str, Any]] = []
        if not retrieved_docs:
            logger.info("Aucun résultat pertinent trouvé pour la requête.")
            return []
        for doc, score in retrieved_docs:
            output_excerpts.append(
                {
                    "text": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score),
                }
            )
        logger.info("Récupéré %d extraits.", len(output_excerpts))
        return output_excerpts

    async def _arun(
        self, query_or_keywords: str, k_retrieval_count: int | None = None
    ) -> list[dict[str, Any]]:
        logger.warning(
            "L'exécution asynchrone de JournalContextRetrieverTool "
            "appelle la version synchrone."
        )
        return self._run(
            query_or_keywords=query_or_keywords, k_retrieval_count=k_retrieval_count
        )
