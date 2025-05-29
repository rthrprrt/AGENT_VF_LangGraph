# src/tools/t1_journal_context_retriever.py
import logging
import os
from typing import (
    Any,
)

# Assurer que Dict et List sont bien ici
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.pydantic_v1 import (
    BaseModel,
    Field,
    PrivateAttr,
)

# Importer PrivateAttr
from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


class JournalContextRetrieverArgs(BaseModel):
    """Input arguments for JournalContextRetrieverTool."""

    query_or_keywords: str = Field(
        description="La requête ou les mots-clés à rechercher dans le journal."
    )
    k_retrieval_count: int | None = Field(default=3, ge=1, le=10)

    class Config:
        arbitrary_types_allowed = True


class JournalContextRetrieverTool(BaseTool):
    name: str = "journal_context_retriever"
    description: str = (
        "Récupère des extraits pertinents et anonymisés du journal d'apprentissage..."
    )
    args_schema: type[BaseModel] = JournalContextRetrieverArgs

    vector_store_path: str
    embedding_model_name: str

    # Utiliser PrivateAttr pour les attributs qui ne font pas partie du schéma public de l'outil
    # et qui sont pour un usage interne/caching.
    _embeddings_model: FastEmbedEmbeddings | None = PrivateAttr(default=None)
    _vector_store: FAISS | None = PrivateAttr(default=None)

    # __init__ n'est pas nécessaire si on utilise PrivateAttr et que les champs
    # publics sont gérés par BaseTool. Si on a besoin d'une logique spécifique
    # à l'initialisation qui n'est pas juste l'assignation des champs, on peut l'ajouter.
    # def __init__(self, **data: Any):
    #     super().__init__(**data)
    #     self._embeddings_model = None # Initialisation explicite
    #     self._vector_store = None   # Initialisation explicite

    def _initialize_dependencies(self) -> bool:
        if self._embeddings_model is None:
            try:
                self._embeddings_model = FastEmbedEmbeddings(
                    model_name=self.embedding_model_name
                )
                logger.info(
                    "Tool T1: FastEmbedEmbeddings initialized with model %s",
                    self.embedding_model_name,
                )
            except Exception as e:  # noqa: BLE001
                logger.error(
                    "Tool T1: Échec init embedding model (%s): %s",
                    self.embedding_model_name,
                    e,
                    exc_info=True,
                )
                return False

        if self._vector_store is None and self._embeddings_model:
            index_file = os.path.join(self.vector_store_path, "index.faiss")
            pkl_file = os.path.join(self.vector_store_path, "index.pkl")

            if not (
                os.path.exists(self.vector_store_path)
                and os.path.exists(index_file)
                and os.path.exists(pkl_file)
            ):
                logger.error(
                    "Tool T1: Vector store non trouvé ou incomplet à %s (manque index.faiss ou index.pkl)",
                    self.vector_store_path,
                )
                return False
            try:
                # Note: allow_dangerous_deserialization=True est important pour FAISS avec pickle
                self._vector_store = FAISS.load_local(
                    self.vector_store_path,
                    self._embeddings_model,
                    allow_dangerous_deserialization=True,
                )
                logger.info(
                    "Tool T1: FAISS vector store loaded from %s", self.vector_store_path
                )
                if (
                    self._vector_store.index is None
                    or self._vector_store.index.ntotal == 0
                ):
                    logger.warning(
                        "Tool T1: L'index FAISS à %s est vide.", self.vector_store_path
                    )
            except Exception as e:  # noqa: BLE001
                logger.error(
                    "Tool T1: Erreur chargement index FAISS %s: %s",
                    self.vector_store_path,
                    e,
                    exc_info=True,
                )
                return False
        return True

    def _perform_similarity_search(self, query: str, k: int) -> list[dict[str, Any]]:
        if (
            not self._vector_store or not self._vector_store.index
        ):  # Ajout de la vérification de .index
            # Essayer d'initialiser si ce n'est pas fait (devrait l'être par _run)
            if (
                not self._initialize_dependencies()
                or not self._vector_store
                or not self._vector_store.index
            ):
                logger.warning(
                    "Tool T1: Vector store non initialisé ou index manquant pour la recherche."
                )
                return [{"error": "Vector store non disponible pour la recherche."}]

        if self._vector_store.index.ntotal == 0:
            logger.warning(
                "Tool T1: L'index FAISS à %s est vide. Recherche impossible.",
                self.vector_store_path,
            )
            return []

        try:
            retrieved_docs_with_scores = (
                self._vector_store.similarity_search_with_score(query, k=k)
            )
            output_excerpts: list[dict[str, Any]] = []
            if not retrieved_docs_with_scores:
                logger.info(
                    "Tool T1: Aucun résultat pertinent trouvé pour la requête: '%s'",
                    query,
                )
                return []

            for doc, score in retrieved_docs_with_scores:
                output_excerpts.append(
                    {
                        "text": doc.page_content,
                        "metadata": doc.metadata,
                        "score": float(score),
                    }
                )
            logger.info(
                "Tool T1: Récupéré %d extraits pour la requête: '%s'",
                len(output_excerpts),
                query,
            )
            return output_excerpts
        except Exception as e:  # noqa: BLE001
            logger.error(
                "Tool T1: Erreur recherche similarité pour '%s': %s",
                query,
                e,
                exc_info=True,
            )
            return [
                {
                    "error": "Erreur lors de la recherche de similarité.",
                    "details": str(e),
                }
            ]

    def _run(
        self, query_or_keywords: str, k_retrieval_count: int | None = None
    ) -> list[dict[str, Any]]:
        logger.info(
            "Tool T1: Récupération de contexte du journal pour : '%s' avec k=%s",
            query_or_keywords,
            k_retrieval_count,
        )

        try:
            args = self.args_schema(  # Utiliser self.args_schema
                query_or_keywords=query_or_keywords, k_retrieval_count=k_retrieval_count
            )
            effective_k = (
                args.k_retrieval_count if args.k_retrieval_count is not None else 3
            )  # Utiliser le défaut du modèle
        except Exception as e_val:  # pydantic.v1.error_wrappers.ValidationError renommé
            logger.error(
                "Tool T1: Erreur de validation des arguments: %s", e_val, exc_info=True
            )
            return [
                {"error": "Arguments invalides pour l'outil T1.", "details": str(e_val)}
            ]

        if not (1 <= effective_k <= 10):  # Validation manuelle pour k
            logger.warning(
                "Tool T1: k_retrieval_count (%d) hors des bornes [1, 10]. Ajustement à 3.",
                effective_k,
            )
            effective_k = 3

        if not self._initialize_dependencies():
            return [
                {"error": "Échec de l'initialisation des dépendances de l'outil T1."}
            ]

        return self._perform_similarity_search(args.query_or_keywords, effective_k)

    async def _arun(
        self, query_or_keywords: str, k_retrieval_count: int | None = None
    ) -> list[dict[str, Any]]:
        logger.warning(
            "Tool T1: L'exécution asynchrone de JournalContextRetrieverTool "
            "appelle la version synchrone _run."
        )
        return self._run(
            query_or_keywords=query_or_keywords, k_retrieval_count=k_retrieval_count
        )
