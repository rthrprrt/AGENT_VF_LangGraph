# src/tools/t1_journal_context_retriever.py
import logging
import os
from typing import Any  # Ajout de List, Optional

from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document  # Ajout de Document
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


class JournalContextRetrieverArgs(BaseModel):
    """Arguments for the JournalContextRetrieverTool."""

    query_or_keywords: str = Field(
        description="La requête ou les mots-clés à rechercher dans le journal."
    )
    k_retrieval_count: int = Field(  # Rendre k_retrieval_count non optionnel ici
        default=3, description="Nombre de chunks pertinents à récupérer (défaut 3)."
    )

    class Config:
        """Configuration pour Pydantic model."""

        arbitrary_types_allowed = True


class JournalContextRetrieverTool(BaseTool):
    """Outil pour récupérer des extraits pertinents du journal d'apprentissage."""

    name: str = "journal_context_retriever"
    description: str = (
        "Récupère des extraits pertinents et anonymisés du journal "
        "d'apprentissage de l'étudiant en fonction d'une requête ou de "
        "mots-clés. Utiliser pour trouver des expériences spécifiques, des "
        "tâches, des réflexions ou des détails mentionnés dans le journal."
    )
    args_schema: type[BaseModel] = JournalContextRetrieverArgs

    vector_store_path: str
    embedding_model_name: str

    def _initialize_embeddings(
        self,
    ) -> FastEmbedEmbeddings | None:  # Changé pour Optional
        """Initializes and returns the embedding model."""
        try:
            return FastEmbedEmbeddings(model_name=self.embedding_model_name)
        except Exception as e:
            logger.error(
                "Échec init embedding model (%s): %s",
                self.embedding_model_name,
                e,
                exc_info=True,
            )
            return None

    def _load_vector_store(
        self, embeddings: FastEmbedEmbeddings
    ) -> FAISS | None:  # Changé pour Optional
        """Loads the FAISS vector store from the configured path."""
        index_file = os.path.join(self.vector_store_path, "index.faiss")
        pkl_file = os.path.join(
            self.vector_store_path, "index.pkl"
        )  # FAISS aussi besoin du .pkl

        # Vérifier l'existence du répertoire et des fichiers essentiels
        if (
            not os.path.isdir(self.vector_store_path)
            or not os.path.exists(index_file)
            or not os.path.exists(pkl_file)
        ):
            logger.error(
                "Vector store non trouvé ou incomplet à %s. Fichiers index.faiss et index.pkl requis.",
                self.vector_store_path,
            )
            return None
        try:
            return FAISS.load_local(
                self.vector_store_path, embeddings, allow_dangerous_deserialization=True
            )
        except FileNotFoundError:  # Gérer spécifiquement si les fichiers ne sont pas trouvés (devrait être couvert par le check os.path.exists)
            logger.error(
                "Fichier index FAISS (index.faiss ou index.pkl) non trouvé dans %s",
                self.vector_store_path,
            )
            return None
        except Exception as e:
            logger.error(
                "Erreur chargement index FAISS %s: %s",
                self.vector_store_path,
                e,
                exc_info=True,
            )
            return None

    def _perform_similarity_search(
        self, vector_store: FAISS, query: str, k: int
    ) -> list[tuple[Document, float]] | None:  # Changé pour Optional et List
        """Performs similarity search on the vector store."""
        if vector_store.index is None or vector_store.index.ntotal == 0:
            logger.warning("L'index FAISS à %s est vide.", self.vector_store_path)
            return []  # Retourner une liste vide si l'index est vide
        try:
            return vector_store.similarity_search_with_score(query, k=k)
        except Exception as e:
            logger.error("Erreur recherche similarité: %s", e, exc_info=True)
            return None

    def _run(
        self,
        query_or_keywords: str,
        k_retrieval_count: int = 3,  # k_retrieval_count a une valeur par défaut ici
    ) -> list[dict[str, Any]]:
        """Main execution logic for the tool."""
        logger.info(
            "Récupération de contexte pour : '%s' avec k=%s",
            query_or_keywords,
            k_retrieval_count,
        )
        # effective_k est maintenant directement k_retrieval_count car il a une valeur par défaut
        effective_k = k_retrieval_count

        embeddings = self._initialize_embeddings()
        if not embeddings:
            return [{"error": "Échec init embedding model", "details": "Voir logs"}]

        vector_store = self._load_vector_store(embeddings)
        if not vector_store:
            error_msg_part1 = "Vector store non trouvé ou erreur de chargement à"
            error_msg_part2 = f" {self.vector_store_path}"
            return [{"error": error_msg_part1 + error_msg_part2}]

        retrieved_docs_with_scores = self._perform_similarity_search(
            vector_store, query_or_keywords, effective_k
        )
        if retrieved_docs_with_scores is None:  # Erreur pendant la recherche
            return [{"error": "Erreur lors de la recherche", "details": "Voir logs"}]

        output_excerpts: list[dict[str, Any]] = []
        if (
            not retrieved_docs_with_scores
        ):  # Liste vide (pas d'erreur, juste pas de résultat)
            logger.info("Aucun résultat pertinent trouvé pour la requête.")
            return []  # Retourner une liste vide

        for doc, score in retrieved_docs_with_scores:
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
        self, query_or_keywords: str, k_retrieval_count: int = 3
    ) -> list[dict[str, Any]]:
        logger.warning(
            "L'exécution asynchrone de JournalContextRetrieverTool "
            "appelle la version synchrone."
        )
        # effective_k est directement k_retrieval_count car il a une valeur par défaut
        return self._run(
            query_or_keywords=query_or_keywords, k_retrieval_count=k_retrieval_count
        )
