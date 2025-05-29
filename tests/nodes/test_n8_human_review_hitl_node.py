# src/nodes/n8_human_review_hitl_node.py
import logging
from typing import Any

# PAS d'import de 'interrupt' ou 'Command' directement ici pour l'instant
# L'interruption sera gérée par la logique du graphe et le checkpointer
from src.state import AgentState, HumanReviewFeedback, SectionDetail, SectionStatus

logger = logging.getLogger(__name__)


class N8HumanReviewHITLNode:
    """
    Node for Human-in-the-Loop review of a generated thesis section draft.
    If no human response is present in the state for the current section,
    it prepares for HITL by populating state.interrupt_payload.
    If a response is present (in section.temporary_human_response), it processes it.
    """

    def run(self, state: AgentState) -> dict[str, Any]:  # noqa: C901
        """
        Prepares for or processes human review feedback for a section.

        Args:
            state: The current state of the agent.

        Returns:
            A dictionary of fields to update in the AgentState.
        """
        logger.info("N8: Human Review HITL Node starting.")
        updated_fields: dict[str, Any] = {
            "current_operation_message": "N8: Evaluating human review state.",
            "interrupt_payload": None,  # Assurer qu'il est vidé s'il n'y a pas d'interruption
            # last_successful_node sera mis à la fin
        }

        current_section_id = state.current_section_id
        current_section_idx_from_state = state.current_section_index

        if current_section_id is None or current_section_idx_from_state is None:
            msg = "N8: Missing current section_id or current_section_index. Cannot proceed."
            logger.error(msg)
            updated_fields["error_message"] = msg
            updated_fields["last_successful_node"] = "N8HumanReviewHITLNode_Error"
            return updated_fields

        target_section: SectionDetail | None = None
        actual_section_index: int | None = None

        thesis_outline_list = (
            state.thesis_outline if isinstance(state.thesis_outline, list) else []
        )

        for i, s_detail in enumerate(thesis_outline_list):
            if s_detail.id == current_section_id:
                target_section = s_detail
                actual_section_index = i
                break

        if target_section is None or actual_section_index is None:
            msg = f"N8: Section with ID '{current_section_id}' not found in thesis_outline."
            logger.error(msg)
            updated_fields["error_message"] = msg
            updated_fields["last_successful_node"] = "N8HumanReviewHITLNode_Error"
            updated_fields["thesis_outline"] = thesis_outline_list
            return updated_fields

        section_to_process = target_section.copy(deep=True)
        human_response_data = section_to_process.temporary_human_response

        if human_response_data:
            logger.info(
                f"N8: Processing human response for section '{section_to_process.title}': {human_response_data}"
            )
            section_to_process.temporary_human_response = None

            action = human_response_data.get("action")
            feedback_text = human_response_data.get("feedback_text")

            if action == "approve_section":
                section_to_process.status = SectionStatus.CONTENT_APPROVED
                section_to_process.final_content = (
                    section_to_process.refined_draft
                    if section_to_process.refined_draft is not None
                    else section_to_process.draft_v1
                )
                section_to_process.human_review_feedback = None
                updated_fields["current_operation_message"] = (
                    f"N8: Section '{section_to_process.title}' approved."
                )
                updated_fields["current_section_index_for_router"] = (
                    current_section_idx_from_state + 1
                )

            elif action == "modify_section":
                if not (
                    feedback_text
                    and isinstance(feedback_text, str)
                    and feedback_text.strip()
                ):
                    section_to_process.status = SectionStatus.HUMAN_REVIEW_PENDING
                    section_to_process.human_review_feedback = HumanReviewFeedback(
                        modification_requested=True,
                        feedback_text="Error: Modification requested but feedback was missing or invalid.",
                    )
                    updated_fields["current_operation_message"] = (
                        f"N8: Modif. for '{section_to_process.title}' had invalid feedback. Awaiting valid review."
                    )
                    updated_fields["current_section_index_for_router"] = (
                        current_section_idx_from_state
                    )
                else:
                    section_to_process.status = SectionStatus.MODIFICATION_REQUESTED
                    section_to_process.human_review_feedback = HumanReviewFeedback(
                        modification_requested=True, feedback_text=feedback_text
                    )
                    updated_fields["current_operation_message"] = (
                        f"N8: Section '{section_to_process.title}' sent for modification."
                    )
                    updated_fields["current_section_index_for_router"] = (
                        current_section_idx_from_state
                    )
            else:
                section_to_process.status = SectionStatus.HUMAN_REVIEW_PENDING
                section_to_process.human_review_feedback = HumanReviewFeedback(
                    modification_requested=True,
                    feedback_text=f"Error: Unknown action '{action}' received.",
                )
                updated_fields["current_operation_message"] = (
                    f"N8: Unknown action '{action}'. Awaiting valid review."
                )
                updated_fields["current_section_index_for_router"] = (
                    current_section_idx_from_state
                )

            updated_fields["last_successful_node"] = "N8HumanReviewHITLNode_Processed"

        else:
            # Première passe : préparer pour l'interruption (sans la lever explicitement ici)
            if section_to_process.status != SectionStatus.HUMAN_REVIEW_PENDING:
                logger.info(
                    f"N8: Section '{section_to_process.title}' status was {section_to_process.status.value}. "
                    "Setting to HUMAN_REVIEW_PENDING."
                )
                section_to_process.status = SectionStatus.HUMAN_REVIEW_PENDING

            draft_to_review = (
                section_to_process.refined_draft
                if section_to_process.refined_draft is not None
                else section_to_process.draft_v1
            )

            if draft_to_review is None:
                msg = f"N8: No draft for section {section_to_process.id} to present for review."
                logger.warning(msg)
                section_to_process.status = SectionStatus.ERROR
                section_to_process.error_details_n6_drafting = (
                    section_to_process.error_details_n6_drafting or ""
                ) + "\nN8 Error: No draft content for review."
                updated_fields["error_message"] = msg
                updated_fields["current_section_index_for_router"] = (
                    current_section_idx_from_state + 1
                )
                updated_fields["last_successful_node"] = "N8HumanReviewHITLNode_Error"
            else:
                # Préparer le payload que l'interface utilisateur externe pourrait lire
                interrupt_payload_for_ui = {
                    "node_name": "N8_HumanReviewHITLNode",
                    "section_id": section_to_process.id,
                    "title": section_to_process.title,
                    "draft_content": draft_to_review,
                    "critique_v1": section_to_process.critique_v1,
                    "instructions": "Review the draft. Provide response with 'action' ('approve_section'/'modify_section') and 'feedback_text' if modifying.",
                }
                logger.info(
                    f"N8: Prepared for human review of section: '{section_to_process.title}'. Payload set in state.interrupt_payload."
                )
                updated_fields["interrupt_payload"] = (
                    interrupt_payload_for_ui  # Stocker pour l'UI
                )
                updated_fields["current_operation_message"] = (
                    f"N8: Section '{section_to_process.title}' is pending human review."
                )
                updated_fields["last_successful_node"] = (
                    "N8HumanReviewHITLNode_Interrupted"
                )
                # Le routeur N4 ne devrait pas avancer l'index si on attend une revue.
                updated_fields["current_section_index_for_router"] = (
                    current_section_idx_from_state
                )

        # Mettre à jour la section dans la liste thesis_outline
        final_thesis_outline = list(thesis_outline_list)
        if actual_section_index < len(final_thesis_outline):
            final_thesis_outline[actual_section_index] = section_to_process
        else:
            logger.error(
                f"N8: actual_section_index {actual_section_index} is out of bounds for thesis_outline (len {len(final_thesis_outline)})."
            )
        updated_fields["thesis_outline"] = final_thesis_outline

        logger.info(
            f"--- FIN NŒUD N8 --- Statut section {current_section_id}: {section_to_process.status.value}"
        )
        return updated_fields
