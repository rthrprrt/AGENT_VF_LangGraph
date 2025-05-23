# tests/test_initial_setup.py
import pytest # Assurez-vous que pytest est une dépendance de dev

from src.state import AgentState
from src.nodes.n0_initial_setup import initialize_agent_settings
from src.config import settings # Pour accéder aux valeurs par défaut attendues


def test_initialize_agent_settings_populates_defaults():
    """
    Tests if initialize_agent_settings correctly populates
    the state with default paths and model names.
    """
    initial_state = AgentState() # Commence avec un état vide ou Pydantic par défaut

    # Exécute la fonction du nœud
    updated_state = initialize_agent_settings(initial_state)

    # Vérifie que les champs attendus sont peuplés
    assert updated_state.school_guidelines_path == settings.default_school_guidelines_path
    assert updated_state.journal_path == settings.default_journal_path
    assert updated_state.output_directory == settings.default_output_directory
    assert updated_state.vector_store_path == settings.vector_store_directory
    assert updated_state.llm_model_name == settings.llm_model_name
    assert updated_state.embedding_model_name == settings.embedding_model_name
    assert updated_state.current_operation_message == "Agent settings initialized."
    assert updated_state.last_successful_node == "N0_InitialSetupNode"

def test_initialize_agent_settings_preserves_overrides():
    """
    Tests if initialize_agent_settings preserves paths if they are
    already set in the initial state.
    """
    custom_guidelines_path = "custom/path/guidelines.pdf"
    custom_journal_path = "custom/journal/"
    initial_state = AgentState(
        school_guidelines_path=custom_guidelines_path,
        journal_path=custom_journal_path
    )

    updated_state = initialize_agent_settings(initial_state)

    assert updated_state.school_guidelines_path == custom_guidelines_path
    assert updated_state.journal_path == custom_journal_path
    # Les autres devraient toujours être des valeurs par défaut si non spécifiées
    assert updated_state.output_directory == settings.default_output_directory