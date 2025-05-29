# src/nodes/n4_section_processor_router.py
import logging
from typing import Any

from src.state import AgentState, SectionStatus

logger = logging.getLogger(__name__)


class N4SectionProcessorRouter:
    """
    Nœud de routage pour déterminer la prochaine étape dans le traitement des sections
    de la thèse.
    """

    def run(self, state: AgentState) -> dict[str, Any]:
        """
        Évalue l'état actuel de `thesis_outline` et détermine le prochain nœud.

        Logique de routage :
        1. Si la section courante a une demande de modification (feedback humain),
           route vers N5 pour re-traiter cette section.
        2. Sinon, cherche la prochaine section avec le statut PENDING à partir de
           l'index de routage actuel.
        3. Si aucune section PENDING n'est trouvée après l'index actuel, cherche
           une section PENDING depuis le début de la liste (au cas où une section
           antérieure serait repassée à PENDING).
        4. Si toutes les sections sont dans un état final (approuvé, erreur, skippé),
           route vers la gestion de la bibliographie (N9).
        5. Gère les cas d'erreur (outline vide, états inconsistants).
        """
        logger.info("N4: Section Processor Router evaluating next step...")
        updated_fields: dict[str, Any] = {
            "current_operation_message": "N4: Deciding next section to process.",
            "next_node_override": None,
        }

        if not state.thesis_outline:
            logger.warning("N4: Thesis outline is empty. Cannot proceed.")
            updated_fields["error_message"] = "N4: Thesis outline is empty."
            updated_fields["next_node_override"] = "ERROR_HANDLER"
            return updated_fields

        start_index = state.current_section_index_for_router
        num_sections = len(state.thesis_outline)

        if (
            0 <= start_index < num_sections
            and state.thesis_outline[start_index].human_review_feedback
            and state.thesis_outline[
                start_index
            ].human_review_feedback.modification_requested
        ):
            current_section = state.thesis_outline[start_index]
            logger.info(
                "N4: Section '%s' (ID: %s) requires modification. "
                "Routing to context retrieval.",
                current_section.title,
                current_section.id,
            )
            updated_fields["current_section_id"] = current_section.id
            updated_fields["current_section_index"] = start_index
            updated_fields["next_node_override"] = "N5_ContextRetrievalNode"
            return updated_fields

        for i in range(start_index, num_sections):
            section = state.thesis_outline[i]
            if section.status == SectionStatus.PENDING:
                logger.info(
                    "N4: Section '%s' (ID: %s) is pending. "
                    "Routing to context retrieval.",
                    section.title,
                    section.id,
                )
                updated_fields["current_section_id"] = section.id
                updated_fields["current_section_index"] = i
                updated_fields["next_node_override"] = "N5_ContextRetrievalNode"
                return updated_fields

        for i in range(start_index):
            section = state.thesis_outline[i]
            if section.status == SectionStatus.PENDING:
                logger.info(
                    "N4: Found earlier PENDING section '%s' (ID: %s). "
                    "Routing to context retrieval.",
                    section.title,
                    section.id,
                )
                updated_fields["current_section_id"] = section.id
                updated_fields["current_section_index"] = i
                updated_fields["current_section_index_for_router"] = i
                updated_fields["next_node_override"] = "N5_ContextRetrievalNode"
                return updated_fields

        all_sections_processed_or_error = all(
            s.status
            in [
                SectionStatus.CONTENT_APPROVED,
                SectionStatus.ERROR,
                SectionStatus.SKIPPED_BY_USER,
            ]
            for s in state.thesis_outline
        )

        if all_sections_processed_or_error:
            logger.info(
                "N4: Toutes les sections semblent traitées ou en erreur. "
                "Routage vers la compilation finale."
            )
            updated_fields["next_node_override"] = "N9_BibliographyManagerNode"
        else:
            logger.warning(
                "N4: No PENDING sections found, but not all sections are "
                "in a final state. Review needed. Defaulting to compile for now."
            )
            updated_fields["next_node_override"] = "N9_BibliographyManagerNode"
            updated_fields["error_message"] = (
                "N4: Inconsistent section states. Check thesis_outline."
            )

        return updated_fields
