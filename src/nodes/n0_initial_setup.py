# src/nodes/n0_initial_setup.py
import logging

from src.config import settings
from src.state import AgentState

logger = logging.getLogger(__name__)


def initialize_agent_settings(state: AgentState) -> AgentState:
    """Initializes paths and basic configurations for the agent.

    This node primarily populates the state with initial path values from config
    if not already provided (e.g., via direct input to the graph).
    """
    logger.info("N0: Initializing agent settings...")

    if state.school_guidelines_path is None:
        state.school_guidelines_path = settings.default_school_guidelines_path
        logger.info(
            f"  Set school_guidelines_path to default: {state.school_guidelines_path}"
        )

    if state.journal_path is None:
        state.journal_path = settings.default_journal_path
        logger.info(f"  Set journal_path to default: {state.journal_path}")

    if state.output_directory == AgentState.model_fields["output_directory"].default:
        state.output_directory = settings.default_output_directory
        logger.info(f"  Set output_directory to default: {state.output_directory}")

    if state.vector_store_path == AgentState.model_fields["vector_store_path"].default:
        state.vector_store_path = settings.vector_store_directory
        logger.info(f"  Set vector_store_path to default: {state.vector_store_path}")

    state.llm_model_name = settings.llm_model_name
    state.embedding_model_name = settings.embedding_model_name

    state.current_operation_message = "Agent settings initialized."
    state.last_successful_node = "N0_InitialSetupNode"
    logger.info("N0: Agent settings initialization complete.")
    return state


if __name__ == "__main__":
    test_state_default = AgentState()
    updated_state = initialize_agent_settings(test_state_default)

    print("\nUpdated State after N0:")
    # Correction E501 pour les prints
    print(f"  School Guidelines Path: {updated_state.school_guidelines_path}")
    print(f"  Journal Path: {updated_state.journal_path}")
    print(f"  Output Directory: {updated_state.output_directory}")
    print(f"  Vector Store Path: {updated_state.vector_store_path}")
    print(f"  LLM Model: {updated_state.llm_model_name}")
    print(f"  Embedding Model: {updated_state.embedding_model_name}")
    print(f"  Message: {updated_state.current_operation_message}")
