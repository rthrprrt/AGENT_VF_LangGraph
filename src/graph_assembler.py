import logging
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from agent_vf_langgraph.state import AgentState
from agent_vf_langgraph.config import settings
from agent_vf_langgraph.nodes.n0_initial_setup import initialize_agent_settings
from agent_vf_langgraph.nodes.n1_guideline_ingestor import ingest_school_guidelines
# Import other nodes as they are created

logger = logging.getLogger(__name__)

def create_graph():
    """
    Creates and configures the LangGraph StateGraph for AGENT_VF_LangGraph.
    """
    logger.info("Creating AGENT_VF_LangGraph...")

    # In-memory checkpointer
    # memory = SqliteSaver.from_conn_string(":memory:") # For quick tests
    
    # Persistent checkpointer
    memory = SqliteSaver.from_conn_string(settings.persistence_db_path)
    logger.info(f"Using SQLiteSaver for persistence: {settings.persistence_db_path}")


    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("N0_InitialSetupNode", initialize_agent_settings)
    workflow.add_node("N1_GuidelineIngestorNode", ingest_school_guidelines)
    # ... add N2_JournalIngestorAndAnonymizerNode etc.

    # Define edges
    workflow.set_entry_point("N0_InitialSetupNode")
    workflow.add_edge("N0_InitialSetupNode", "N1_GuidelineIngestorNode")
    
    # For now, a simple end after N1 for testing setup
    workflow.add_edge("N1_GuidelineIngestorNode", END) 

    # TODO: Define conditional edges and routing logic
    # e.g., workflow.add_conditional_edges(...)

    app = workflow.compile(checkpointer=memory)
    logger.info("AGENT_VF_LangGraph compiled successfully.")
    return app

# Example of running the graph (for testing)
if __name__ == "__main__":
    import uuid
    
    # Ensure directories from config.py exist if nodes try to access them
    import os
    os.makedirs(settings.default_output_directory, exist_ok=True)
    os.makedirs(settings.vector_store_directory, exist_ok=True)
    os.makedirs("data/input/school_guidelines/", exist_ok=True) # For dummy file if needed
    
    # Create a dummy guideline file for N1 to pick up if it were to read a file
    # For now, N1 uses placeholder text, so this is not strictly needed for current N1
    # dummy_guideline_path = settings.default_school_guidelines_path
    # if not os.path.exists(dummy_guideline_path):
    #     with open(dummy_guideline_path, "w") as f:
    #         f.write("This is a dummy guideline PDF content.")
    #     logger.info(f"Created dummy guideline file: {dummy_guideline_path}")


    graph_app = create_graph()

    # Define an initial state or inputs
    # If the graph relies on specific input paths, ensure they are valid or mocked
    initial_input = AgentState(
        # school_guidelines_path="path/to/your/guidelines.pdf", # Can override defaults
        # journal_path="path/to/your/journal_entries_directory/"
    )
    
    # Configuration for a run
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    logger.info(f"Invoking graph with thread_id: {thread_id}")
    
    final_state = None
    try:
        for event in graph_app.stream(initial_input, config=config, stream_mode="values"):
            # 'event' here will be the full AgentState after each node completes
            logger.info(f"Current state after node: {event.last_successful_node if event else 'Unknown'}")
            final_state = event # Keep the last state

        if final_state:
            logger.info("\n--- Final State ---")
            logger.info(f"School Guidelines Raw Text (first 100 chars): {final_state.school_guidelines_raw_text[:100] if final_state.school_guidelines_raw_text else 'None'}...")
            logger.info(f"School Guidelines Structured: {final_state.school_guidelines_structured}")
            logger.info(f"School Guidelines Formatting: {final_state.school_guidelines_formatting}")
            logger.info(f"Error Message: {final_state.error_message}")
            logger.info(f"Last Successful Node: {final_state.last_successful_node}")
        else:
            logger.warning("Graph did not produce a final state.")

    except Exception as e:
        logger.error(f"Error running the graph: {e}", exc_info=True)

    # To inspect intermediate states with time travel (if needed):
    # snapshots = graph_app.get_state_history(config)
    # for i, snapshot in enumerate(snapshots):
    # print(f"\n--- Snapshot {i} ---")
    # print(snapshot.values)