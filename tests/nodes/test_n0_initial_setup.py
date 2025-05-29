# tests/nodes/test_n0_initial_setup.py
from pathlib import Path
from unittest.mock import patch

from src.config import Settings
from src.nodes.n0_initial_setup import N0InitialSetupNode
from src.state import AgentState


def test_initialize_agent_settings_populates_defaults(tmp_path):
    """Test that default settings are populated correctly."""
    node = N0InitialSetupNode()
    initial_state = AgentState(
        output_directory=str(tmp_path / "outputs"),
        vector_store_path=str(tmp_path / "vector_store"),
        # Laisser les autres champs à None pour qu'ils prennent les valeurs de mock_settings_instance
        llm_model_name=None,
        embedding_model_name=None,
        recreate_vector_store=AgentState.__fields__[
            "recreate_vector_store"
        ].default,  # Utiliser le défaut du modèle
        school_guidelines_path=None,
        journal_path=None,
    )

    mock_settings_instance = Settings(
        llm_model_name="default_llm_from_settings",
        embedding_model_name="default_embedding_from_settings",
        recreate_vector_store=False,
        default_school_guidelines_path=(
            "data/input/school_guidelines/Mission_Professionnelle_Digi5_EPITECH.pdf"
        ),
        default_journal_path="data/input/journal_entries/",
        default_output_directory="outputs/theses_default_settings",
        vector_store_directory="data/processed/vector_store_default_settings",
        k_retrieval_count=5,  # Doit être défini dans Settings
    )

    with patch("src.nodes.n0_initial_setup.settings", mock_settings_instance):
        result = node.run(initial_state)

    assert (
        result["school_guidelines_path"]
        == mock_settings_instance.default_school_guidelines_path
    )
    assert result["journal_path"] == mock_settings_instance.default_journal_path
    # output_directory et vector_store_path sont fournis dans initial_state, donc ils ne sont pas écrasés
    assert result["output_directory"] == str(tmp_path / "outputs")
    assert result["vector_store_path"] == str(tmp_path / "vector_store")
    assert result["llm_model_name"] == "default_llm_from_settings"
    assert result["embedding_model_name"] == "default_embedding_from_settings"
    assert (
        result["recreate_vector_store"] is False
    )  # Car initial_state.recreate_vector_store prend le défaut du modèle (False)
    # et state_dict_set_fields ne le voit pas comme "set"
    # donc settings.recreate_vector_store (False) est utilisé.
    assert Path(result["output_directory"]).exists()
    assert Path(result["vector_store_path"]).exists()
    user_persona_field = AgentState.__fields__.get("user_persona")
    assert result["user_persona"] == (
        user_persona_field.default if user_persona_field else None
    )
    assert result["last_successful_node"] == "N0InitialSetupNode"


def test_initialize_agent_settings_preserves_overrides(tmp_path):
    """Test that settings in AgentState override defaults from Settings."""
    custom_output_dir = tmp_path / "custom" / "outputs_alt"
    custom_vector_dir = tmp_path / "custom" / "vector_store_alt"

    initial_state = AgentState(
        school_guidelines_path="custom/guidelines.pdf",
        journal_path="custom/journal/",
        output_directory=str(custom_output_dir),
        vector_store_path=str(custom_vector_dir),
        llm_model_name="custom_llm",
        embedding_model_name="custom_embedding",
        recreate_vector_store=True,  # Explicitement True
        user_persona="custom_persona",
        example_thesis_text_content="some example content",
    )
    node = N0InitialSetupNode()

    mock_settings_instance = Settings(
        default_school_guidelines_path="SHOULD_BE_OVERRIDDEN",
        default_journal_path="SHOULD_BE_OVERRIDDEN",
        default_output_directory="SHOULD_BE_OVERRIDDEN",
        vector_store_directory="SHOULD_BE_OVERRIDDEN",
        llm_model_name="SHOULD_BE_OVERRIDDEN",
        embedding_model_name="SHOULD_BE_OVERRIDDEN",
        recreate_vector_store=False,  # Cette valeur de settings sera ignorée car recreate_vector_store est set dans initial_state
        k_retrieval_count=5,
    )
    with patch("src.nodes.n0_initial_setup.settings", mock_settings_instance):
        result = node.run(initial_state)

    assert result["school_guidelines_path"] == "custom/guidelines.pdf"
    assert result["journal_path"] == "custom/journal/"
    assert result["output_directory"] == str(custom_output_dir)
    assert result["vector_store_path"] == str(custom_vector_dir)
    assert Path(result["output_directory"]).exists()
    assert Path(result["vector_store_path"]).exists()
    assert result["llm_model_name"] == "custom_llm"
    assert result["embedding_model_name"] == "custom_embedding"
    assert result["recreate_vector_store"] is True  # Doit rester True de initial_state
    assert result["user_persona"] == "custom_persona"
    assert result["example_thesis_text_content"] == "some example content"
    assert result["last_successful_node"] == "N0InitialSetupNode"
