# src/graph_assembler.py
import logging
import os  # Assurez-vous que os est importé
import uuid  # Assurez-vous que uuid est importé

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from src.config import settings

# Assurez-vous que les imports des nœuds sont corrects
from src.nodes.n0_initial_setup import initialize_agent_settings
from src.nodes.n1_guideline_ingestor import ingest_school_guidelines
from src.state import AgentState

logger = logging.getLogger(__name__)


def create_graph():
    """Creates/configures the LangGraph StateGraph for AGENT_VF_LangGraph."""
    # La docstring ci-dessus est maintenant sur une seule ligne.

    logger.info("Creating AGENT_VF_LangGraph...")

    memory = SqliteSaver.from_conn_string(settings.persistence_db_path)
    # Ligne potentiellement problématique (était graph_assembler.py:21:89)
    logger.info(
        f"Using SQLiteSaver for persistence: {settings.persistence_db_path}"
    )  # Déjà coupée, devrait être OK.

    workflow = StateGraph(AgentState)

    workflow.add_node("N0_InitialSetupNode", initialize_agent_settings)
    workflow.add_node("N1_GuidelineIngestorNode", ingest_school_guidelines)

    workflow.set_entry_point("N0_InitialSetupNode")
    workflow.add_edge("N0_InitialSetupNode", "N1_GuidelineIngestorNode")
    workflow.add_edge("N1_GuidelineIngestorNode", END)

    app = workflow.compile(checkpointer=memory)
    logger.info("AGENT_VF_LangGraph compiled successfully.")
    return app


if __name__ == "__main__":
    os.makedirs(settings.default_output_directory, exist_ok=True)
    os.makedirs(settings.vector_store_directory, exist_ok=True)
    os.makedirs("data/input/school_guidelines/", exist_ok=True)

    graph_app = create_graph()
    initial_input = AgentState()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    logger.info(f"Invoking graph with thread_id: {thread_id}")

    final_state = None
    try:
        for event in graph_app.stream(
            initial_input, config=config, stream_mode="values"
        ):
            # Ligne potentiellement problématique (était graph_assembler.py:80:89)
            log_message_node = event.last_successful_node if event else "Unknown"
            logger.info(f"Current state after node: {log_message_node}")
            final_state = event

        if final_state:
            logger.info("\n--- Final State ---")
            raw_text_preview = (
                final_state.school_guidelines_raw_text[:100]
                if final_state.school_guidelines_raw_text
                else "None"
            )
            # Ligne potentiellement problématique (était graph_assembler.py:96:89)
            logger.info(
                "School Guidelines Raw Text (first 100 chars): "
                f"{raw_text_preview}..."
            )
            logger.info(
                "School Guidelines Structured: "
                f"{final_state.school_guidelines_structured}"
            )
            # Ligne potentiellement problématique (était graph_assembler.py:99:89)
            logger.info(
                "School Guidelines Formatting: "
                f"{final_state.school_guidelines_formatting}"
            )
            logger.info(f"Error Message: {final_state.error_message}")
            logger.info(f"Last Successful Node: {final_state.last_successful_node}")
        else:
            logger.warning("Graph did not produce a final state.")

    except Exception:
        logger.error("Error running the graph", exc_info=True)
