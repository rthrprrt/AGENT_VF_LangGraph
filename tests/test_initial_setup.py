# Imports triés : d'abord les modules de votre projet (src)
from src.config import settings
from src.nodes.n0_initial_setup import initialize_agent_settings
from src.state import AgentState


def test_initialize_agent_settings_populates_defaults():
    """Tests if initialize_agent_settings correctly populates defaults."""
    # Pas besoin d'une ligne vide ici si la docstring est sur une seule ligne
    initial_state = AgentState()

    updated_state = initialize_agent_settings(initial_state)

    # Les asserts sont corrects pour les tests. S101 sera ignoré via pyproject.toml.
    # Lignes coupées pour respecter la longueur maximale si nécessaire,
    # mais les asserts simples tiennent généralement bien.
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
    """Tests if initialize_agent_settings preserves overridden paths."""
    custom_guidelines_path = "custom/path/guidelines.pdf"
    custom_journal_path = "custom/journal/"
    initial_state = AgentState(
        school_guidelines_path=custom_guidelines_path,
        journal_path=custom_journal_path,
    )

    updated_state = initialize_agent_settings(initial_state)

    assert updated_state.school_guidelines_path == custom_guidelines_path
    assert updated_state.journal_path == custom_journal_path
    # Vérifie aussi qu'un champ non surchargé prend bien la valeur par défaut
    assert updated_state.output_directory == settings.default_output_directory


# Assurez-vous que ce fichier se termine par une nouvelle ligne (W292).
# `ruff format .` devrait s'en charger.
