# tests/nodes/test_n6_section_drafting.py
import logging
import unittest
from unittest.mock import patch
from uuid import uuid4

from langchain_core.messages import AIMessage

from src.config import settings
from src.nodes.n6_section_drafting import N6SectionDraftingNode
from src.state import (
    AgentState,
    SectionDetail,
    SectionStatus,
)

logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.DEBUG) # Décommenter pour debug si besoin


class TestN6SectionDraftingNode(unittest.TestCase):
    def setUp(self):
        """Set up a basic AgentState and N6 node for testing."""
        self.node = N6SectionDraftingNode()

        self.section_id_1 = str(uuid4())
        self.section_id_2 = str(uuid4())
        self.initial_state_dict = {
            "llm_model_name": settings.llm_model_name,
            "user_persona": "Test Persona: Étudiant en Master IA",
            "thesis_outline": [
                SectionDetail(
                    id=self.section_id_1,
                    title="1. Introduction Test",
                    level=1,
                    description_objectives="Objectifs de l'introduction.",
                    original_requirements_summary="Résumé des exigences pour l'intro.",
                    key_questions_to_answer=[
                        "Quelle est la problématique ?",
                        "Quel est le plan ?",
                    ],
                    example_phrasing_or_content_type="Style formel et engageant.",
                    anonymized_context_for_llm=(
                        "Contexte du journal: Projet Alpha réalisé. "
                        "Difficulté X surmontée."
                    ),
                    status=SectionStatus.CONTEXT_RETRIEVED,
                    # Initialiser les nouveaux champs pour N6/N7
                    critique_v1=None,
                    current_draft_for_critique=None,
                    reflection_attempts=0,
                    reflection_history=[],
                ),
                SectionDetail(
                    id=self.section_id_2,
                    title="2. Chapitre Suivant",
                    level=1,
                    description_objectives="Objectifs du chapitre suivant.",
                    original_requirements_summary="Exigences pour le chapitre suivant.",
                    key_questions_to_answer=["Question A?", "Question B?"],
                    anonymized_context_for_llm=None,
                    status=SectionStatus.CONTEXT_RETRIEVED,
                    critique_v1=None,
                    current_draft_for_critique=None,
                    reflection_attempts=0,
                    reflection_history=[],
                ),
            ],
            "current_section_id": self.section_id_1,
            "current_section_index": 0,
            "max_reflection_attempts": 3,  # Ajouté car référencé dans AgentState
        }
        self.state = AgentState(**self.initial_state_dict)

    @patch("src.nodes.n6_section_drafting.ChatOllama")
    def test_run_drafts_section_successfully(self, mock_chat_ollama):
        """Test nominal: N6 drafts a section based on context and plan."""
        mock_llm_instance = mock_chat_ollama.return_value
        mock_llm_instance.invoke.return_value = AIMessage(
            content="Contenu rédigé pour l'introduction."
        )
        self.node.llm = mock_llm_instance

        # S'assurer qu'il n'y a pas de critique pour forcer le mode initial draft
        target_section = self.state.get_section_by_id(self.section_id_1)
        assert target_section is not None
        target_section.critique_v1 = None
        target_section.status = (
            SectionStatus.CONTEXT_RETRIEVED
        )  # Statut avant N6 initial

        updated_state_fields = self.node.run(self.state)

        mock_llm_instance.invoke.assert_called_once()
        assert updated_state_fields.get("error_message") is None
        assert (
            updated_state_fields.get("last_successful_node") == "N6SectionDraftingNode"
        )
        # Correction de l'assertion du message
        expected_message = (
            "N6: Initial draft generated for section '1. Introduction Test'."
        )
        assert updated_state_fields.get("current_operation_message") == expected_message

        updated_outline = updated_state_fields.get("thesis_outline")
        assert updated_outline is not None
        if updated_outline:
            drafted_section = updated_outline[0]
            assert drafted_section.id == self.section_id_1
            assert drafted_section.draft_v1 == "Contenu rédigé pour l'introduction."
            assert drafted_section.status == SectionStatus.DRAFT_GENERATED
            # Vérifier que current_draft_for_critique est bien mis à jour
            assert (
                drafted_section.current_draft_for_critique
                == "Contenu rédigé pour l'introduction."
            )

    @patch("src.nodes.n6_section_drafting.ChatOllama")
    def test_run_handles_no_journal_context(self, mock_chat_ollama):
        """Test drafting when anonymized_context_for_llm is None or empty."""
        self.state.current_section_id = self.section_id_2
        self.state.current_section_index = 1

        section_without_context = self.state.get_section_by_id(
            self.state.current_section_id
        )
        assert section_without_context is not None
        section_without_context.anonymized_context_for_llm = ""
        section_without_context.critique_v1 = None  # Assurer mode initial draft
        section_without_context.status = SectionStatus.CONTEXT_RETRIEVED

        mock_llm_instance = mock_chat_ollama.return_value
        mock_llm_instance.invoke.return_value = AIMessage(
            content="Contenu rédigé sans contexte journal spécifique."
        )
        self.node.llm = mock_llm_instance

        updated_state_fields = self.node.run(self.state)

        mock_llm_instance.invoke.assert_called_once()
        args, _ = mock_llm_instance.invoke.call_args
        assert "[Aucun extrait de journal spécifique" in args[0].to_string()

        assert updated_state_fields.get("error_message") is None
        updated_outline = updated_state_fields.get("thesis_outline")
        assert updated_outline is not None
        if updated_outline:
            drafted_section = updated_outline[1]
            assert (
                drafted_section.draft_v1
                == "Contenu rédigé sans contexte journal spécifique."
            )
            assert drafted_section.status == SectionStatus.DRAFT_GENERATED
            assert (
                drafted_section.current_draft_for_critique
                == "Contenu rédigé sans contexte journal spécifique."
            )

    @patch("src.nodes.n6_section_drafting.ChatOllama")
    def test_run_handles_llm_invocation_error(self, mock_chat_ollama):
        """Test N6 handles exceptions during LLM call."""
        mock_llm_instance = mock_chat_ollama.return_value
        mock_llm_instance.invoke.side_effect = Exception("Simulated LLM Error")
        self.node.llm = mock_llm_instance

        target_section = self.state.get_section_by_id(self.section_id_1)
        assert target_section is not None
        target_section.critique_v1 = None
        target_section.status = SectionStatus.CONTEXT_RETRIEVED

        updated_state_fields = self.node.run(self.state)

        assert updated_state_fields.get("error_message") is not None
        assert "Error during LLM call or processing" in updated_state_fields.get(
            "error_message", ""
        )

        updated_outline = updated_state_fields.get("thesis_outline")
        assert updated_outline is not None
        if updated_outline:
            error_section = updated_outline[0]
            assert error_section.status == SectionStatus.ERROR
            assert "Simulated LLM Error" in (
                error_section.error_details_n6_drafting or ""
            )

    def test_run_no_current_section_id(self):
        """Test N6 handles missing current_section_id."""
        self.state.current_section_id = None
        updated_state_fields = self.node.run(self.state)
        assert updated_state_fields.get("error_message") is not None
        assert "current_section_id is not set" in updated_state_fields.get(
            "error_message", ""
        )

    def test_run_section_not_found_in_outline(self):
        """Test N6 handles if current_section_id is not in the outline."""
        self.state.current_section_id = "id_inexistant"
        updated_state_fields = self.node.run(self.state)
        assert updated_state_fields.get("error_message") is not None
        assert "not found in thesis_outline" in updated_state_fields.get(
            "error_message", ""
        )

    @patch("src.nodes.n6_section_drafting.ChatOllama", None)
    def test_run_llm_not_initialized(self):
        """Test N6 handles case where LLM failed to initialize."""
        node_with_failed_llm = N6SectionDraftingNode()
        node_with_failed_llm.llm = None

        updated_state_fields = node_with_failed_llm.run(self.state)
        assert updated_state_fields.get("error_message") is not None
        assert "LLM not initialized" in updated_state_fields.get("error_message", "")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
