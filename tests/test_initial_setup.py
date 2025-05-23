# tests/test_initial_setup.py
# import pytest # F401 - Supprimé car non utilisé directement ici

from src.config import settings # settings en premier (import de votre code)
from src.nodes.n0_initial_setup import initialize_agent_settings
from src.state import AgentState


def test_initialize_agent_settings_populates_defaults():
    """
    Tests if initialize_agent_settings correctly populates the state
    with default paths and model names.
    """
    # D205 corrigé avec une ligne vide
    initial_state = AgentState()

    updated_state = initialize_agent_settings(initial_state)

    # E501 corrigé en coupant la ligne d'assert
    assert (
        updated_state.school_guidelines_path
        == settings.default_school_guidelines_path
    )
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
    # D205 corrigé avec une ligne vide
    custom_guidelines_path = "custom/path/guidelines.pdf"
    custom_journal_path = "custom/journal/"
    initial_state = AgentState(
        school_guidelines_path=custom_guidelines_path,
        journal_path=custom_journal_path,
    )

    updated_state = initialize_agent_settings(initial_state)

    assert updated_state.school_guidelines_path == custom_guidelines_path
    assert updated_state.journal_path == custom_journal_path
    assert updated_state.output_directory == settings.default_output_directory
# W292 sera corrigé par ruff format . en ajoutant une newline à la fin