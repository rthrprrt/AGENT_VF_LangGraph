import logging
from agent_vf_langgraph.state import AgentState
from agent_vf_langgraph.config import settings # Import global settings

logger = logging.getLogger(__name__)

def initialize_agent_settings(state: AgentState) -> AgentState:
    """
    Initializes paths and basic configurations for the agent.
    This node primarily populates the state with initial path values from config
    if not already provided (e.g., via direct input to the graph).
    """
    logger.info("N0: Initializing agent settings...")

    # Ensure output directories exist if not already handled elsewhere
    # os.makedirs(settings.default_output_directory, exist_ok=True)
    # os.makedirs(settings.vector_store_directory, exist_ok=True)

    # Set paths from config if not overridden in initial state
    if state.school_guidelines_path is None:
        state.school_guidelines_path = settings.default_school_guidelines_path
        logger.info(f"  Set school_guidelines_path to default: {state.school_guidelines_path}")

    if state.journal_path is None:
        state.journal_path = settings.default_journal_path
        logger.info(f"  Set journal_path to default: {state.journal_path}")
    
    if state.output_directory == "outputs/theses" : # Default from pydantic
        state.output_directory = settings.default_output_directory
        logger.info(f"  Set output_directory to default: {state.output_directory}")

    if state.vector_store_path == "data/processed/vector_store": # Default from pydantic
        state.vector_store_path = settings.vector_store_directory
        logger.info(f"  Set vector_store_path to default: {state.vector_store_path}")

    # Set model names from config
    state.llm_model_name = settings.llm_model_name
    state.embedding_model_name = settings.embedding_model_name
    
    state.current_operation_message = "Agent settings initialized."
    state.last_successful_node = "N0_InitialSetupNode"
    logger.info("N0: Agent settings initialization complete.")
    return state

# Example usage (for testing the node standalone)
if __name__ == "__main__":
    initial_state = AgentState()
    # You could override paths here for testing:
    # initial_state.school_guidelines_path = "custom/path/to/guidelines.pdf"
    
    updated_state = initialize_agent_settings(initial_state)
    
    print("Updated State after N0:")
    print(f"  School Guidelines Path: {updated_state.school_guidelines_path}")
    print(f"  Journal Path: {updated_state.journal_path}")
    print(f"  Output Directory: {updated_state.output_directory}")
    print(f"  Vector Store Path: {updated_state.vector_store_path}")
    print(f"  LLM Model: {updated_state.llm_model_name}")
    print(f"  Embedding Model: {updated_state.embedding_model_name}")
    print(f"  Message: {updated_state.current_operation_message}")