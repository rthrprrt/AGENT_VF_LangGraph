# tests/nodes/test_n8_human_review_hitl_node.py
import logging
import unittest

from src.config import settings

# Garder MagicMock pour compatibilité si utilisé ailleurs
from src.nodes.n8_human_review_hitl_node import N8HumanReviewHITLNode
from src.state import (
    AgentState,
    CritiqueOutput,  # Importer CritiqueOutput
    SectionDetail,
    SectionStatus,
)

logger = logging.getLogger(__name__)


class TestN8HumanReviewHITLNode(unittest.TestCase):
    """Tests for the N8HumanReviewHITLNode."""

    def setUp(self):
        """Set up a basic AgentState and N8 node for testing."""
        self.node = N8HumanReviewHITLNode()
        self.section_id_1 = "section_1_id_n8"
        self.initial_draft_content = "Ceci est un brouillon initial pour la section 1."
        # self.critique_content = "Critique V1 pour le brouillon." # Ancienne chaîne
        # Nouvelle initialisation pour critique_v1 avec un objet CritiqueOutput ou None
        # Pour les tests de N8, nous n'avons pas besoin d'une critique complexe ici,
        # car N8 se concentre sur le payload d'interruption et le traitement de la réponse humaine.
        # Le contenu de la critique est plus pertinent pour N7.
        # Si un test spécifique a besoin d'une critique, il peut la créer.
        self.mock_critique_v1_content = CritiqueOutput(
            overall_assessment_score=3,
            overall_assessment_summary="Critique N7 simulée pour N8.",
            identified_flaws=[],
            missing_information=[],
            superfluous_content=[],
            suggested_search_queries=[],
            final_recommendation="REVISION_NEEDED",  # Ou une autre valeur selon le test
        )

        self.refined_draft_content = "Ceci est un brouillon raffiné après critique."

        self.state_dict = {
            "llm_model_name": settings.llm_model_name,
            "user_persona": "Test Persona N8",
            "thesis_outline": [
                SectionDetail(
                    id=self.section_id_1,
                    title="1. Section pour N8",
                    level=1,
                    description_objectives="Objectifs N8.",
                    original_requirements_summary="Req N8.",
                    draft_v1=self.initial_draft_content,
                    # critique_v1=self.critique_content, # Ancien
                    critique_v1=self.mock_critique_v1_content,  # Initialiser avec un mock CritiqueOutput
                    refined_draft=None,
                    status=SectionStatus.SELF_CRITIQUE_COMPLETED,
                    # Assurer que les nouveaux champs de SectionDetail sont présents avec des valeurs par défaut
                    current_draft_for_critique=self.initial_draft_content,  # Pourrait être le draft_v1
                    reflection_history=[],
                    reflection_attempts=0,
                ),
                SectionDetail(
                    id="section_2_id_n8",
                    title="2. Autre Section N8",
                    level=1,
                    description_objectives="Obj N8-2",
                    original_requirements_summary="Req N8-2",
                    draft_v1="Brouillon pour section 2",
                    status=SectionStatus.DRAFT_GENERATED,
                    critique_v1=None,  # Initialiser à None pour cette section non ciblée
                    current_draft_for_critique="Brouillon pour section 2",
                    reflection_history=[],
                    reflection_attempts=0,
                ),
            ],
            "current_section_id": self.section_id_1,
            "current_section_index": 0,
            "current_section_index_for_router": 0,
            "interrupt_payload": None,
            "error_message": None,
            "max_reflection_attempts": 3,  # Ajouté car utilisé par AgentState
        }
        self.current_state = AgentState(**self.state_dict)

    def test_run_first_pass_prepares_interrupt_payload(self):
        """N8: Si pas de réponse humaine, prépare interrupt_payload et met statut."""
        section_in_state = self.current_state.get_section_by_id(self.section_id_1)
        assert section_in_state is not None
        section_in_state.temporary_human_response = None
        section_in_state.status = SectionStatus.SELF_CRITIQUE_COMPLETED
        # S'assurer que current_draft_for_critique est bien le draft_v1 pour ce test
        section_in_state.current_draft_for_critique = section_in_state.draft_v1
        section_in_state.refined_draft = (
            None  # S'assurer qu'il n'y a pas de refined_draft
        )

        updated_fields = self.node.run(self.current_state)

        assert updated_fields.get("error_message") is None
        payload = updated_fields.get("interrupt_payload")
        assert payload is not None
        assert payload["section_id"] == self.section_id_1
        assert (
            payload["draft_content"] == self.initial_draft_content
        )  # Doit être draft_v1
        # Vérifier que critique_v1 dans le payload est bien l'objet CritiqueOutput mocké
        assert (
            payload["critique_v1"] == self.mock_critique_v1_content.dict()
        )  # Comparer les dictionnaires

        updated_outline = updated_fields.get("thesis_outline", [])
        processed_section = updated_outline[0]
        assert processed_section.status == SectionStatus.HUMAN_REVIEW_PENDING
        assert (
            updated_fields.get("last_successful_node")
            == "N8HumanReviewHITLNode_Interrupted"
        )
        assert updated_fields.get("current_section_index_for_router") == 0

    def test_run_first_pass_uses_refined_draft_if_available(self):
        """N8: interrupt_payload doit utiliser refined_draft si présent."""
        section_in_state = self.current_state.get_section_by_id(self.section_id_1)
        assert section_in_state is not None
        section_in_state.refined_draft = self.refined_draft_content
        # current_draft_for_critique devrait être le refined_draft s'il existe
        section_in_state.current_draft_for_critique = self.refined_draft_content
        section_in_state.temporary_human_response = None
        section_in_state.status = (
            SectionStatus.SELF_CRITIQUE_COMPLETED
        )  # Ou un statut après révision N6

        updated_fields = self.node.run(self.current_state)
        payload = updated_fields.get("interrupt_payload")
        assert payload is not None
        assert payload["draft_content"] == self.refined_draft_content

    def test_run_processes_approve_action_from_state(self):
        """N8: Traite une action d'approbation humaine."""
        section_in_state = self.current_state.get_section_by_id(self.section_id_1)
        assert section_in_state is not None
        section_in_state.temporary_human_response = {"action": "approve_section"}
        section_in_state.refined_draft = self.refined_draft_content
        section_in_state.current_draft_for_critique = self.refined_draft_content

        updated_fields = self.node.run(self.current_state)

        assert updated_fields.get("interrupt_payload") is None
        updated_outline = updated_fields.get("thesis_outline", [])
        processed_section = updated_outline[0]

        assert processed_section.status == SectionStatus.CONTENT_APPROVED
        assert processed_section.final_content == self.refined_draft_content
        assert processed_section.human_review_feedback is None
        assert (
            updated_fields.get("last_successful_node")
            == "N8HumanReviewHITLNode_Processed"
        )
        assert (
            updated_fields.get("current_section_index_for_router")
            == self.current_state.current_section_index + 1
        )

    def test_run_processes_modify_action_from_state(self):
        """N8: Traite une demande de modification humaine."""
        feedback = "Veuillez développer davantage le point X."
        section_in_state = self.current_state.get_section_by_id(self.section_id_1)
        assert section_in_state is not None
        section_in_state.temporary_human_response = {
            "action": "modify_section",
            "feedback_text": feedback,
        }
        section_in_state.current_draft_for_critique = (
            section_in_state.draft_v1
        )  # Ou refined_draft s'il existait

        updated_fields = self.node.run(self.current_state)

        assert updated_fields.get("interrupt_payload") is None
        updated_outline = updated_fields.get("thesis_outline", [])
        processed_section = updated_outline[0]

        assert processed_section.status == SectionStatus.MODIFICATION_REQUESTED
        assert processed_section.human_review_feedback is not None
        assert processed_section.human_review_feedback.modification_requested is True
        assert processed_section.human_review_feedback.feedback_text == feedback
        assert (
            updated_fields.get("last_successful_node")
            == "N8HumanReviewHITLNode_Processed"
        )
        assert (
            updated_fields.get("current_section_index_for_router")
            == self.current_state.current_section_index
        )

    def test_run_no_draft_content_available(self):
        """N8: Gère l'absence de brouillon à reviewer."""
        section_in_state = self.current_state.get_section_by_id(self.section_id_1)
        assert section_in_state is not None
        section_in_state.draft_v1 = None
        section_in_state.refined_draft = None
        section_in_state.current_draft_for_critique = None  # Important pour le test
        section_in_state.temporary_human_response = None

        updated_fields = self.node.run(self.current_state)

        assert updated_fields.get("interrupt_payload") is None
        error_msg = updated_fields.get("error_message", "")
        assert f"N8: No draft for section {self.section_id_1}" in error_msg
        updated_outline = updated_fields.get("thesis_outline", [])
        processed_section = updated_outline[0]
        assert processed_section.status == SectionStatus.ERROR
        assert "N8 Error: No draft content for review" in (
            processed_section.error_details_n6_drafting or ""
        )  # Doit être n6 ou un champ N8
        assert (
            updated_fields.get("last_successful_node") == "N8HumanReviewHITLNode_Error"
        )

    def test_run_no_current_section_id(self):
        """N8: Gère l'absence de current_section_id."""
        self.current_state.current_section_id = None
        self.current_state.current_section_index = None
        updated_fields = self.node.run(self.current_state)
        expected_msg = (
            "N8: Missing current section_id or current_section_index. "
            "Cannot proceed."
        )
        assert updated_fields.get("error_message") == expected_msg
        assert (
            updated_fields.get("last_successful_node") == "N8HumanReviewHITLNode_Error"
        )

    def test_run_section_not_found(self):
        """N8: Gère si current_section_id ne correspond à aucune section."""
        bad_id = "id_non_existant_n8"
        self.current_state.current_section_id = bad_id
        updated_fields = self.node.run(self.current_state)
        expected_msg = f"N8: Section with ID '{bad_id}' not found in thesis_outline."
        assert updated_fields.get("error_message", "") == expected_msg
        assert (
            updated_fields.get("last_successful_node") == "N8HumanReviewHITLNode_Error"
        )
