# src/nodes/n5_context_retrieval.py
import logging
from typing import Any

from src.config import settings

# Importations réelles
from src.state import AgentState, SectionDetail, SectionStatus
from src.tools.t1_journal_context_retriever import (
    JournalContextRetrieverArgs,
    JournalContextRetrieverTool,
)

logger = logging.getLogger(__name__)


class N5ContextRetrievalNode:
    """
    Nœud responsable de la récupération du contexte du journal pour une section de thèse donnée.
    """

    def _construct_query_from_keywords(self, keywords: list[str]) -> str:
        if not keywords:
            return ""
        return "Expériences, apprentissages et réflexions liés à : " + ", ".join(
            keywords
        )

    def run(self, state: AgentState) -> dict[str, Any]:
        logger.info(
            "--- EXÉCUTION DU NŒUD N5 : RÉCUPÉRATION DE CONTEXTE DU JOURNAL ---"
        )
        updated_fields: dict[str, Any] = {
            "last_successful_node": "N5_ContextRetrievalNode",
            "current_operation_message": "Initialisation de la récupération de contexte.",
            "error_message": None,
        }

        current_section_id = state["current_section_id"]  # Accès par clé
        thesis_outline = state["thesis_outline"]  # Accès par clé

        if not current_section_id:
            logger.error("current_section_id non trouvé dans l'état.")
            updated_fields["error_message"] = (
                "ID de section courant manquant pour la récupération de contexte."
            )
            updated_fields["current_operation_message"] = (
                "Échec : ID de section courant manquant."
            )
            return updated_fields

        section_to_update: SectionDetail | None = None
        section_index: int | None = None

        for i, section_detail_item in enumerate(
            thesis_outline
        ):  # Renommer 'section' pour éviter conflit
            if section_detail_item.id == current_section_id:
                section_to_update = section_detail_item
                section_index = i
                break

        if section_to_update is None or section_index is None:
            logger.error(
                f"SectionDetail avec ID {current_section_id} non trouvée dans thesis_outline."
            )
            updated_fields["error_message"] = (
                f"Section {current_section_id} non trouvée."
            )
            updated_fields["current_operation_message"] = (
                f"Échec : Section {current_section_id} non trouvée."
            )
            return updated_fields

        logger.info(
            f"Traitement de la section : {section_to_update.title} (ID: {current_section_id})"
        )

        keywords = section_to_update.student_experience_keywords

        if not keywords:
            logger.warning(
                f"Aucun student_experience_keywords pour la section {current_section_id}."
            )
            updated_fields["current_operation_message"] = (
                f"Aucun mot-clé pour la section {current_section_id}."
            )
            section_to_update.retrieved_journal_excerpts = []
            section_to_update.anonymized_context_for_llm = ""
            section_to_update.status = SectionStatus.CONTEXT_RETRIEVED
        else:
            query_str = self._construct_query_from_keywords(keywords)
            logger.info(f"Requête pour T1 : '{query_str}'")

            try:
                retriever_tool = JournalContextRetrieverTool(
                    vector_store_path=state["vector_store_path"],  # Accès par clé
                    embedding_model_name=settings.embedding_model_name,
                )

                k = settings.k_retrieval_count

                # L'appel à l'outil T1 se ferait via un ToolNode dans le graphe.
                # Pour la logique interne du noeud N5, on peut imaginer que le résultat de T1
                # serait disponible (par exemple, via un appel direct si ce noeud gérait aussi l'exécution de l'outil,
                # ou via l'état si un ToolNode précédent a été appelé).
                # Ici, nous allons simuler la réception de la sortie de T1.
                # Pour les tests unitaires, nous mockerons _run de l'outil.
                # Dans la version intégrée, N5 recevrait la sortie de T1 (via l'état, après appel par un ToolNode)
                # ou instancierait et appellerait l'outil. L'instanciation dans run est une approche.

                # En supposant que le Nœud N5 est celui qui appelle l'outil (via une bibliothèque comme ToolExecutor ou directement)
                tool_args = JournalContextRetrieverArgs(
                    query_or_keywords=query_str, k_retrieval_count=k
                )
                raw_excerpts = retriever_tool._run(**tool_args.dict())

                if any(
                    isinstance(excerpt, dict) and "error" in excerpt
                    for excerpt in raw_excerpts
                ):
                    error_detail = next(
                        (
                            e.get("details", "Erreur inconnue")
                            for e in raw_excerpts
                            if "error" in e
                        ),
                        "Erreur inconnue",
                    )
                    logger.error(
                        f"L'outil T1 a retourné une erreur pour {current_section_id}: {error_detail}"
                    )
                    updated_fields["error_message"] = (
                        f"Erreur de l'outil T1 : {error_detail}"
                    )
                    section_to_update.status = SectionStatus.ERROR_CONTEXT_RETRIEVAL
                elif not raw_excerpts:
                    logger.info(
                        f"Aucun extrait pertinent trouvé par T1 pour {current_section_id}."
                    )
                    section_to_update.retrieved_journal_excerpts = []
                    section_to_update.anonymized_context_for_llm = ""
                    section_to_update.status = SectionStatus.CONTEXT_RETRIEVED
                else:
                    section_to_update.retrieved_journal_excerpts = raw_excerpts
                    anonymized_context_for_llm = "\n\n---\n\n".join(
                        [
                            excerpt["text"]
                            for excerpt in raw_excerpts
                            if "text" in excerpt
                        ]
                    )
                    section_to_update.anonymized_context_for_llm = (
                        anonymized_context_for_llm
                    )
                    section_to_update.status = SectionStatus.CONTEXT_RETRIEVED
                    logger.info(
                        f"{len(raw_excerpts)} extraits récupérés pour {current_section_id}."
                    )

                updated_fields["current_operation_message"] = (
                    f"Récupération de contexte pour section {current_section_id} terminée. Statut: {section_to_update.status.value}"
                )

            except Exception as e:
                logger.exception(
                    f"Exception lors de la récupération de contexte pour {current_section_id}: {e}"
                )
                updated_fields["error_message"] = f"Exception T1 : {str(e)}"
                section_to_update.status = SectionStatus.ERROR_CONTEXT_RETRIEVAL

        new_thesis_outline = list(thesis_outline)
        new_thesis_outline[section_index] = section_to_update
        updated_fields["thesis_outline"] = new_thesis_outline

        logger.info(
            f"--- FIN NŒUD N5 --- Statut section {current_section_id}: {section_to_update.status.value}"
        )
        return updated_fields
