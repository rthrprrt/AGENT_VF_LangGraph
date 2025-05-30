# src/nodes/n8_human_review_hitl_node.py
import logging
from typing import Any

from src.state import AgentState, HumanReviewFeedback, SectionDetail, SectionStatus

logger = logging.getLogger(__name__)


class N8HumanReviewHITLNode:
    """
    Node for Human-in-the-Loop review of a generated thesis section draft.

    If no human response is present in the state for the current section,
    it prepares for HITL by populating state.interrupt_payload.
    If a response is present (in section.temporary_human_response), it processes it.
    """

    def _process_human_approval(
        self,
        section_to_process: SectionDetail,
        current_section_idx_from_state: int,
        updated_fields: dict[str, Any],
    ):
        """Processes the 'approve_section' action from human review."""
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

    def _process_human_modification_request(
        self,
        section_to_process: SectionDetail,
        feedback_text: str | None,
        current_section_idx_from_state: int,
        updated_fields: dict[str, Any],
    ):
        """Processes the 'modify_section' action from human review."""
        if not (
            feedback_text and isinstance(feedback_text, str) and feedback_text.strip()
        ):
            section_to_process.status = SectionStatus.HUMAN_REVIEW_PENDING
            section_to_process.human_review_feedback = HumanReviewFeedback(
                modification_requested=True,
                feedback_text=(
                    "Error: Modification requested but feedback was "
                    "missing or invalid."
                ),
            )
            updated_fields["current_operation_message"] = (
                f"N8: Modif. for '{section_to_process.title}' had invalid "
                f"feedback. Awaiting valid review."
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

    def _handle_unknown_action(
        self,
        section_to_process: SectionDetail,
        action: str | None,
        current_section_idx_from_state: int,
        updated_fields: dict[str, Any],
    ):
        """Handles unknown actions received during human review."""
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

    def _prepare_for_human_review(  # noqa: C901
        self,
        section_to_process: SectionDetail,
        current_section_idx_from_state: int,
        updated_fields: dict[str, Any],
    ):
        """Prepares the payload for human review and sets section status."""
        if section_to_process.status != SectionStatus.HUMAN_REVIEW_PENDING:
            logger.info(
                f"N8: Section '{section_to_process.title}' status was "
                f"{section_to_process.status.value}. Setting to HUMAN_REVIEW_PENDING."
            )
            section_to_process.status = SectionStatus.HUMAN_REVIEW_PENDING

        draft_to_review = (
            section_to_process.refined_draft
            if section_to_process.refined_draft is not None
            else section_to_process.draft_v1
        )

        if draft_to_review is None:
            msg = (
                f"N8: No draft for section {section_to_process.id} to present "
                f"for review."
            )
            logger.warning(msg)
            section_to_process.status = SectionStatus.ERROR
            error_details = section_to_process.error_details_n6_drafting or ""
            section_to_process.error_details_n6_drafting = (
                f"{error_details}\nN8 Error: No draft content for review."
            )
            updated_fields["error_message"] = msg
            updated_fields["current_section_index_for_router"] = (
                current_section_idx_from_state + 1
            )
            updated_fields["last_successful_node"] = "N8HumanReviewHITLNode_Error"
        else:
            interrupt_payload_for_ui = {
                "node_name": "N8_HumanReviewHITLNode",
                "section_id": section_to_process.id,
                "title": section_to_process.title,
                "draft_content": draft_to_review,
                "critique_v1": section_to_process.critique_v1,
                "instructions": (
                    "Review the draft. Provide response with 'action' "
                    "('approve_section'/'modify_section') and 'feedback_text' "
                    "if modifying."
                ),
            }
            logger.info(
                "N8: Prepared for human review of section: '%s'. "
                "Payload set in state.interrupt_payload.",
                section_to_process.title,
            )
            updated_fields["interrupt_payload"] = interrupt_payload_for_ui
            updated_fields["current_operation_message"] = (
                f"N8: Section '{section_to_process.title}' is pending human review."
            )
            updated_fields["last_successful_node"] = "N8HumanReviewHITLNode_Interrupted"
            updated_fields["current_section_index_for_router"] = (
                current_section_idx_from_state
            )

    def run(self, state: AgentState) -> dict[str, Any]:  # noqa: C901
        """
        Prepares for or processes human review feedback for a section.
        """
        logger.info("N8: Human Review HITL Node starting.")
        updated_fields: dict[str, Any] = {
            "current_operation_message": "N8: Evaluating human review state.",
            "interrupt_payload": None,
        }

        current_section_id = state.current_section_id
        current_section_idx = state.current_section_index

        if current_section_id is None or current_section_idx is None:
            # Correction du message d'erreur pour correspondre au test
            msg = (
                "N8: Missing current section_id or current_section_index. "
                "Cannot proceed."
            )
            logger.error(msg)
            updated_fields["error_message"] = msg
            updated_fields["last_successful_node"] = "N8HumanReviewHITLNode_Error"
            return updated_fields

        thesis_outline_list = (
            state.thesis_outline if isinstance(state.thesis_outline, list) else []
        )
        target_section_tuple = next(
            (
                (i, s)
                for i, s in enumerate(thesis_outline_list)
                if s.id == current_section_id
            ),
            (None, None),
        )
        actual_section_index, target_section = target_section_tuple

        if target_section is None or actual_section_index is None:
            # Correction du message d'erreur pour correspondre au test
            msg = f"N8: Section with ID '{current_section_id}' not found in thesis_outline."
            logger.error(msg)
            updated_fields["error_message"] = msg
            updated_fields["last_successful_node"] = "N8HumanReviewHITLNode_Error"
            updated_fields["thesis_outline"] = thesis_outline_list
            return updated_fields

        section_to_process = target_section.copy(deep=True)
        human_response = section_to_process.temporary_human_response

        if human_response:
            logger.info(
                "N8: Processing human response for section '%s': %s",
                section_to_process.title,
                human_response,
            )
            section_to_process.temporary_human_response = None
            action = human_response.get("action")
            feedback = human_response.get("feedback_text")

            if action == "approve_section":
                self._process_human_approval(
                    section_to_process, current_section_idx, updated_fields
                )
            elif action == "modify_section":
                self._process_human_modification_request(
                    section_to_process, feedback, current_section_idx, updated_fields
                )
            else:
                self._handle_unknown_action(
                    section_to_process, action, current_section_idx, updated_fields
                )
            updated_fields["last_successful_node"] = "N8HumanReviewHITLNode_Processed"
        else:
            self._prepare_for_human_review(
                section_to_process, current_section_idx, updated_fields
            )

        final_thesis_outline = list(thesis_outline_list)
        if actual_section_index < len(final_thesis_outline):
            final_thesis_outline[actual_section_index] = section_to_process
        else:  # pragma: no cover
            logger.error(
                "N8: actual_section_index %d out of bounds for outline (len %d).",
                actual_section_index,
                len(final_thesis_outline),
            )
        updated_fields["thesis_outline"] = final_thesis_outline

        logger.info(
            "--- FIN NŒUD N8 --- Statut section %s: %s",
            current_section_id,
            section_to_process.status.value,
        )
        return updated_fields
