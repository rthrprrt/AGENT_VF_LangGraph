# tests/nodes/test_n6_section_drafting.py
import logging
import unittest
from unittest.mock import patch
from uuid import uuid4

from langchain_core.messages import AIMessage

from src.config import settings  # Pour LLM model name
from src.nodes.n6_section_drafting import N6SectionDraftingNode
from src.state import AgentState, SectionDetail, SectionStatus

# Configurer le logging pour voir les messages du nœud pendant les tests si nécessaire
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TestN6SectionDraftingNode(unittest.TestCase):
    def setUp(self):
        """Set up a basic AgentState and N6 node for testing."""
        self.node = N6SectionDraftingNode()  # Le LLM sera mocké dans les tests

        self.section_id_1 = str(uuid4())
        self.initial_state_dict = {
            "llm_model_name": settings.llm_model_name,  # Assurer que N6 peut le lire
            "user_persona": "Test Persona: Étudiant en Master IA",
            "thesis_outline": [
                SectionDetail(
                    id=self.section_id_1,
                    title="1. Introduction Test",
                    level=1,
                    description_objectives="Objectifs de l'introduction.",
                    original_requirements_summary="Résumé des exigences Epitech pour l'intro.",
                    key_questions_to_answer=[
                        "Quelle est la problématique ?",
                        "Quel est le plan ?",
                    ],
                    example_phrasing_or_content_type="Style formel et engageant.",
                    anonymized_context_for_llm="Contexte du journal: Projet Alpha réalisé. Difficulté X surmontée.",
                    status=SectionStatus.CONTEXT_RETRIEVED,  # Statut attendu avant N6
                ),
                SectionDetail(
                    id=str(uuid4()),
                    title="2. Chapitre Suivant",
                    level=1,
                    description_objectives="Objectifs du chapitre suivant.",
                    original_requirements_summary="Exigences pour le chapitre suivant.",
                    key_questions_to_answer=["Question A?", "Question B?"],
                    anonymized_context_for_llm=None,  # Pas de contexte pour ce test
                    status=SectionStatus.CONTEXT_RETRIEVED,
                ),
            ],
            "current_section_id": self.section_id_1,  # Cible la première section
            "current_section_index": 0,  # Index correspondant
        }
        self.state = AgentState(**self.initial_state_dict)

    @patch("src.nodes.n6_section_drafting.ChatOllama")
    def test_run_drafts_section_successfully(self, MockChatOllama):
        """Test nominal: N6 drafts a section based on context and plan."""
        mock_llm_instance = MockChatOllama.return_value
        mock_llm_instance.invoke.return_value = AIMessage(
            content="Contenu rédigé pour l'introduction."
        )

        # Réassigner le mock au nœud car __init__ est appelé avant le patch dans setUp
        self.node.llm = mock_llm_instance

        updated_state_fields = self.node.run(self.state)

        mock_llm_instance.invoke.assert_called_once()
        # On pourrait vérifier des éléments spécifiques dans le prompt passé au LLM
        # args, kwargs = mock_llm_instance.invoke.call_args
        # self.assertIn("Projet Alpha réalisé", args[0].to_string()) # Exemple

        self.assertIsNone(updated_state_fields.get("error_message"))
        self.assertEqual(
            updated_state_fields.get("last_successful_node"), "N6SectionDraftingNode"
        )
        self.assertIn(
            "Draft generated for section '1. Introduction Test'",
            updated_state_fields.get("current_operation_message"),
        )

        updated_outline = updated_state_fields.get("thesis_outline")
        self.assertIsNotNone(updated_outline)
        if updated_outline:
            drafted_section = updated_outline[0]  # La première section a été traitée
            self.assertEqual(drafted_section.id, self.section_id_1)
            self.assertEqual(
                drafted_section.draft_v1, "Contenu rédigé pour l'introduction."
            )
            self.assertEqual(drafted_section.status, SectionStatus.DRAFT_GENERATED)

    @patch("src.nodes.n6_section_drafting.ChatOllama")
    def test_run_handles_no_journal_context(self, MockChatOllama):
        """Test drafting when anonymized_context_for_llm is None or empty."""
        self.state.current_section_id = self.state.thesis_outline[
            1
        ].id  # Cible la section sans contexte
        self.state.current_section_index = 1

        section_without_context = self.state.get_section_by_id(
            self.state.current_section_id
        )
        assert section_without_context is not None
        section_without_context.anonymized_context_for_llm = ""  # ou None

        mock_llm_instance = MockChatOllama.return_value
        mock_llm_instance.invoke.return_value = AIMessage(
            content="Contenu rédigé sans contexte journal spécifique."
        )
        self.node.llm = mock_llm_instance

        updated_state_fields = self.node.run(self.state)

        mock_llm_instance.invoke.assert_called_once()
        args, _ = mock_llm_instance.invoke.call_args
        # Vérifier que le prompt indique l'absence de contexte
        self.assertIn("[Aucun extrait de journal spécifique", args[0].to_string())

        self.assertIsNone(updated_state_fields.get("error_message"))
        updated_outline = updated_state_fields.get("thesis_outline")
        self.assertIsNotNone(updated_outline)
        if updated_outline:
            drafted_section = updated_outline[1]
            self.assertEqual(
                drafted_section.draft_v1,
                "Contenu rédigé sans contexte journal spécifique.",
            )
            self.assertEqual(drafted_section.status, SectionStatus.DRAFT_GENERATED)

    @patch("src.nodes.n6_section_drafting.ChatOllama")
    def test_run_handles_llm_invocation_error(self, MockChatOllama):
        """Test N6 handles exceptions during LLM call."""
        mock_llm_instance = MockChatOllama.return_value
        mock_llm_instance.invoke.side_effect = Exception("Simulated LLM Error")
        self.node.llm = mock_llm_instance

        updated_state_fields = self.node.run(self.state)

        self.assertIsNotNone(updated_state_fields.get("error_message"))
        self.assertIn(
            "Error during LLM call or processing",
            updated_state_fields.get("error_message"),
        )

        updated_outline = updated_state_fields.get("thesis_outline")
        self.assertIsNotNone(updated_outline)
        if updated_outline:
            error_section = updated_outline[0]
            self.assertEqual(error_section.status, SectionStatus.ERROR)
            self.assertIn(
                "Simulated LLM Error", error_section.error_details_n6_drafting or ""
            )

    def test_run_no_current_section_id(self):
        """Test N6 handles missing current_section_id."""
        self.state.current_section_id = None
        updated_state_fields = self.node.run(self.state)
        self.assertIsNotNone(updated_state_fields.get("error_message"))
        self.assertIn(
            "current_section_id is not set", updated_state_fields.get("error_message")
        )

    def test_run_section_not_found_in_outline(self):
        """Test N6 handles if current_section_id is not in the outline."""
        self.state.current_section_id = "id_inexistant"
        updated_state_fields = self.node.run(self.state)
        self.assertIsNotNone(updated_state_fields.get("error_message"))
        self.assertIn(
            "not found in thesis_outline", updated_state_fields.get("error_message")
        )

    @patch(
        "src.nodes.n6_section_drafting.ChatOllama", None
    )  # Simule l'échec d'init du LLM
    def test_run_llm_not_initialized(self):
        """Test N6 handles case where LLM failed to initialize."""
        # Recréer le noeud pour que __init__ soit appelé SANS le mock global de ChatOllama
        node_with_failed_llm = N6SectionDraftingNode()
        # Forcer self.llm à None si l'exception dans __init__ n'a pas déjà fait cela
        node_with_failed_llm.llm = None

        updated_state_fields = node_with_failed_llm.run(self.state)
        self.assertIsNotNone(updated_state_fields.get("error_message"))
        self.assertIn("LLM not initialized", updated_state_fields.get("error_message"))


if __name__ == "__main__":
    unittest.main()
