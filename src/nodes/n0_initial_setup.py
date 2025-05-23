# src/nodes/n0_initial_setup.py
import logging
import os

from src.config import settings
from src.state import AgentState

logger = logging.getLogger(__name__)


def initialize_agent_settings(state: AgentState) -> AgentState:
    """
    Initialize paths and model configurations for the agent from settings.

    This node ensures that essential paths (for guidelines, journal, output,
    vector store) and model names (LLM, embeddings) are populated in the
    agent's state at the beginning of a run. It uses default values from
    the `settings` object if not already overridden in the initial state.

    Args:
        state: Current agent state to initialize

    Returns:
        Updated agent state with initialized settings
    """
    logger.info("N0: Initializing agent settings...")

    if state.school_guidelines_path is None:
        state.school_guidelines_path = settings.default_school_guidelines_path
        logger.info(
            "  Set school_guidelines_path to default: %s",
            state.school_guidelines_path,
        )

    if state.journal_path is None:
        state.journal_path = settings.default_journal_path
        logger.info("  Set journal_path to default: %s", state.journal_path)

    # Utiliser model_fields pour Pydantic V2
    pydantic_default_output_dir = AgentState.model_fields["output_directory"].default
    if state.output_directory == pydantic_default_output_dir:
        state.output_directory = settings.default_output_directory
        logger.info("  Set output_directory to default: %s", state.output_directory)
    else:
        logger.info("  Using provided output_directory: %s", state.output_directory)
    os.makedirs(state.output_directory, exist_ok=True)

    # Utiliser model_fields pour Pydantic V2
    pydantic_default_vector_store_path = AgentState.model_fields[
        "vector_store_path"
    ].default
    if state.vector_store_path == pydantic_default_vector_store_path:
        state.vector_store_path = settings.vector_store_directory
        logger.info("  Set vector_store_path to default: %s", state.vector_store_path)
    else:
        logger.info("  Using provided vector_store_path: %s", state.vector_store_path)

    state.llm_model_name = settings.llm_model_name
    state.embedding_model_name = settings.embedding_model_name
    logger.info("  Set LLM model to: %s", state.llm_model_name)
    logger.info("  Set embedding model to: %s", state.embedding_model_name)

    state.current_operation_message = "Agent settings initialized."
    state.last_successful_node = "N0_InitialSetupNode"
    logger.info("N0: Agent settings initialization complete.")
    return state


if __name__ == "__main__":
    # Test du nœud
    test_state = AgentState()

    updated_state = initialize_agent_settings(test_state)

    print("\nUpdated State after N0:")
    print(f"  School Guidelines Path: {updated_state.school_guidelines_path}")
    print(f"  Journal Path: {updated_state.journal_path}")
    print(f"  Output Directory: {updated_state.output_directory}")
    print(f"  Vector Store Path: {updated_state.vector_store_path}")
    print(f"  LLM Model: {updated_state.llm_model_name}")
    print(f"  Embedding Model: {updated_state.embedding_model_name}")
    print(f"  Message: {updated_state.current_operation_message}")
