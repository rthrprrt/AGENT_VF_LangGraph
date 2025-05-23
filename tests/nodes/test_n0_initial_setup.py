# tests/nodes/test_n0_initial_setup.py
from unittest.mock import patch  # Standard library import

from src.config import settings  # Local application/library specific imports
from src.nodes.n0_initial_setup import initialize_agent_settings
from src.state import AgentState


def test_initialize_agent_settings_populates_defaults():
    """Tests if initialize_agent_settings correctly populates defaults."""
    initial_state = AgentState()

    # Mock os.makedirs to avoid creating directories during test
    with patch("os.makedirs") as mock_makedirs:
        updated_state = initialize_agent_settings(initial_state)
        mock_makedirs.assert_called_once_with(
            settings.default_output_directory, exist_ok=True
        )

    assert (
        updated_state.school_guidelines_path == settings.default_school_guidelines_path
    )
    assert updated_state.journal_path == settings.default_journal_path
    assert updated_state.output_directory == settings.default_output_directory
    assert updated_state.vector_store_path == settings.vector_store_directory
    assert updated_state.llm_model_name == settings.llm_model_name
    assert updated_state.embedding_model_name == settings.embedding_model_name
    assert updated_state.current_operation_message == "Agent settings initialized."
    assert updated_state.last_successful_node == "N0_InitialSetupNode"


def test_initialize_agent_settings_preserves_overrides():
    """Tests if initialize_agent_settings preserves paths if already set."""
    custom_guidelines_path = "custom/path/guidelines.pdf"
    custom_journal_path = "custom/journal/"
    custom_output_dir = "custom/outputs/"
    custom_vector_store_path = "custom/vector_store/"

    initial_state = AgentState(
        school_guidelines_path=custom_guidelines_path,
        journal_path=custom_journal_path,
        output_directory=custom_output_dir,
        vector_store_path=custom_vector_store_path,
    )
    # Mock os.makedirs
    with patch("os.makedirs") as mock_makedirs:
        updated_state = initialize_agent_settings(initial_state)
        mock_makedirs.assert_called_once_with(custom_output_dir, exist_ok=True)

    assert updated_state.school_guidelines_path == custom_guidelines_path
    assert updated_state.journal_path == custom_journal_path
    assert updated_state.output_directory == custom_output_dir
    assert updated_state.vector_store_path == custom_vector_store_path
    # Model names should still be set from settings
    assert updated_state.llm_model_name == settings.llm_model_name
    assert updated_state.embedding_model_name == settings.embedding_model_name
