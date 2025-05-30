# src/nodes/n5_context_retrieval.py
import logging
from typing import Any

from src.config import settings
from src.state import AgentState, SectionDetail, SectionStatus
from src.tools.t1_journal_context_retriever import (
    JournalContextRetrieverArgs,
    JournalContextRetrieverTool,
)

logger = logging.getLogger(__name__)


class N5ContextRetrievalNode:
    """Node for retrieving journal context for a given thesis section."""

    def _construct_query_from_keywords(self, keywords: list[str]) -> str:
        """Constructs a search query from a list of keywords."""
        if not keywords:
            return ""
        return "Expériences, apprentissages et réflexions liés à : " + ", ".join(
            keywords
        )

    def run(self, state: AgentState) -> dict[str, Any]:  # noqa: C901
        """Executes the context retrieval process for the current section."""
        logger.info(
            "--- EXÉCUTION DU NŒUD N5 : RÉCUPÉRATION DE CONTEXTE DU JOURNAL ---"
        )
        updated_fields: dict[str, Any] = {
            "last_successful_node": "N5_ContextRetrievalNode",
            "current_operation_message": "Initialisation de la récupération de contexte.",
            "error_message": None,
        }

        current_section_id = state.current_section_id
        thesis_outline = state.thesis_outline

        if not current_section_id:
            logger.error("N5 Error: current_section_id non trouvé dans l'état.")
            updated_fields["error_message"] = (
                "ID de section courant manquant pour la récupération de contexte."
            )
            updated_fields["current_operation_message"] = (
                "Échec : ID de section courant manquant."
            )
            updated_fields["last_successful_node"] = "N5_ContextRetrievalNode_Error"
            return updated_fields

        section_to_update: SectionDetail | None = None
        section_index: int | None = None

        current_thesis_outline = (
            thesis_outline if isinstance(thesis_outline, list) else []
        )

        for i, section_detail_item in enumerate(current_thesis_outline):
            if section_detail_item.id == current_section_id:
                section_to_update = section_detail_item
                section_index = i
                break

        if section_to_update is None or section_index is None:
            logger.error(
                f"N5 Error: SectionDetail avec ID {current_section_id} non trouvée."
            )
            updated_fields["error_message"] = (
                f"Section {current_section_id} non trouvée."
            )
            updated_fields["current_operation_message"] = (
                f"Échec : Section {current_section_id} non trouvée."
            )
            updated_fields["last_successful_node"] = "N5_ContextRetrievalNode_Error"
            updated_fields["thesis_outline"] = current_thesis_outline
            return updated_fields

        section_copy = section_to_update.copy(deep=True)

        logger.info(
            f"Traitement de la section : {section_copy.title} (ID: {current_section_id})"
        )

        keywords = section_copy.student_experience_keywords

        if not keywords:
            logger.warning(
                f"Aucun student_experience_keywords pour section {current_section_id}."
            )
            updated_fields["current_operation_message"] = (
                f"Aucun mot-clé pour la section {current_section_id}."
            )
            section_copy.retrieved_journal_excerpts = []
            section_copy.anonymized_context_for_llm = (
                "[Aucun mot-clé fourni, donc aucun contexte de journal récupéré.]"
            )
            section_copy.status = SectionStatus.CONTEXT_RETRIEVED
        else:
            query_str = self._construct_query_from_keywords(keywords)
            logger.info(f"Requête pour T1 : '{query_str}'")

            try:
                if not state.vector_store_path:
                    raise ValueError("Vector store path n'est pas défini dans l'état.")

                retriever_tool = JournalContextRetrieverTool(
                    vector_store_path=state.vector_store_path,
                    embedding_model_name=settings.embedding_model_name,
                )
                k = settings.k_retrieval_count
                tool_args = JournalContextRetrieverArgs(
                    query_or_keywords=query_str, k_retrieval_count=k
                )
                raw_excerpts = retriever_tool._run(**tool_args.model_dump())

                if any(
                    isinstance(excerpt, dict) and "error" in excerpt
                    for excerpt in raw_excerpts
                ):
                    error_detail = next(
                        (
                            e.get("details", "Erreur inconnue de T1")
                            for e in raw_excerpts
                            if isinstance(e, dict) and "error" in e
                        ),
                        "Erreur inconnue de T1",
                    )
                    logger.error(
                        f"L'outil T1 a retourné une erreur pour {current_section_id}: "
                        f"{error_detail}"
                    )
                    updated_fields["error_message"] = (
                        f"Erreur de l'outil T1 : {error_detail}"
                    )
                    section_copy.status = SectionStatus.ERROR_CONTEXT_RETRIEVAL
                    section_copy.error_details_n5_context = f"T1 error: {error_detail}"

                elif not raw_excerpts:
                    logger.info(
                        f"Aucun extrait pertinent trouvé par T1 pour {current_section_id}."
                    )
                    section_copy.retrieved_journal_excerpts = []
                    section_copy.anonymized_context_for_llm = "[Aucun extrait de journal pertinent trouvé pour les mots-clés.]"
                    section_copy.status = SectionStatus.CONTEXT_RETRIEVED
                else:
                    section_copy.retrieved_journal_excerpts = raw_excerpts
                    anonymized_context_for_llm = "\n\n---\n\n".join(
                        [
                            excerpt["text"]
                            for excerpt in raw_excerpts
                            if isinstance(excerpt, dict) and "text" in excerpt
                        ]
                    )
                    section_copy.anonymized_context_for_llm = anonymized_context_for_llm
                    section_copy.status = SectionStatus.CONTEXT_RETRIEVED
                    logger.info(
                        f"{len(raw_excerpts)} extraits récupérés pour "
                        f"{current_section_id}."
                    )

                updated_fields["current_operation_message"] = (
                    f"Récupération de contexte pour section {current_section_id} "
                    f"terminée. Statut: {section_copy.status.value}"
                )

            except Exception as e:
                logger.exception(
                    f"Exception lors de la récupération de contexte pour "
                    f"{current_section_id}: {e}"
                )
                updated_fields["error_message"] = f"Exception T1/N5 : {str(e)}"
                section_copy.status = SectionStatus.ERROR_CONTEXT_RETRIEVAL
                section_copy.error_details_n5_context = f"N5 Exception: {str(e)}"
                updated_fields["last_successful_node"] = "N5_ContextRetrievalNode_Error"

        new_thesis_outline = [s.copy(deep=True) for s in current_thesis_outline]
        if section_index is not None:  # Vérification pour mypy
            new_thesis_outline[section_index] = section_copy
        updated_fields["thesis_outline"] = new_thesis_outline

        if (
            not updated_fields.get("error_message")
            and updated_fields.get("last_successful_node")
            != "N5_ContextRetrievalNode_Error"
        ):
            updated_fields["last_successful_node"] = "N5_ContextRetrievalNode"

        logger.info(
            f"--- FIN NŒUD N5 --- Statut section {current_section_id}: "
            f"{section_copy.status.value}"
        )
        return updated_fields
