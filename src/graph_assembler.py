# src/graph_assembler.py
import logging
import os
import uuid  # Ajout pour éviter une potentielle erreur si utilisé plus tard

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from src.config import settings  # Retiré logger de cet import pour éviter confusion
from src.nodes.n0_initial_setup import initialize_agent_settings
from src.nodes.n1_guideline_ingestor import ingest_school_guidelines
from src.nodes.n2_journal_ingestor_anonymizer import (
    journal_ingestor_anonymizer_node,
)
from src.state import AgentState

logger = logging.getLogger(__name__)  # Utiliser un logger local au module


def create_graph(checkpointer_path: str = settings.persistence_db_path):
    """Creates and configures the LangGraph StateGraph for AGENT_VF_LangGraph."""
    logger.info("Creating AGENT_VF_LangGraph workflow...")

    if checkpointer_path == ":memory:":
        memory = SqliteSaver.from_conn_string(":memory:")
        logger.info("Using in-memory SqliteSaver for persistence.")
    else:
        db_dir = os.path.dirname(checkpointer_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        memory = SqliteSaver.from_conn_string(checkpointer_path)
        logger.info("Using SQLiteSaver for persistence: %s", checkpointer_path)

    workflow = StateGraph(AgentState)

    workflow.add_node("N0_InitialSetupNode", initialize_agent_settings)
    workflow.add_node("N1_GuidelineIngestorNode", ingest_school_guidelines)
    workflow.add_node(
        "N2_JournalIngestorAnonymizerNode", journal_ingestor_anonymizer_node
    )

    workflow.set_entry_point("N0_InitialSetupNode")
    workflow.add_edge("N0_InitialSetupNode", "N1_GuidelineIngestorNode")
    workflow.add_edge("N1_GuidelineIngestorNode", "N2_JournalIngestorAnonymizerNode")
    workflow.add_edge("N2_JournalIngestorAnonymizerNode", END)

    app = workflow.compile(checkpointer=memory)
    logger.info("AGENT_VF_LangGraph workflow compiled successfully.")
    return app


if __name__ == "__main__":
    os.makedirs(settings.default_output_directory, exist_ok=True)
    guidelines_dir = os.path.dirname(settings.default_school_guidelines_path)
    if guidelines_dir:
        os.makedirs(guidelines_dir, exist_ok=True)
    journal_dir = settings.default_journal_path
    if journal_dir:
        os.makedirs(journal_dir, exist_ok=True)

    if not os.path.exists(settings.default_school_guidelines_path):
        logger.info(
            "Creating dummy guideline file for testing: %s",
            settings.default_school_guidelines_path,
        )
        with open(settings.default_school_guidelines_path, "w", encoding="utf-8") as f:
            f.write("Ceci est un document de directives factice.\n")
            f.write("Section: Introduction\nContenu de l'intro.\n")
            f.write("Format: Times New Roman 12, interligne 1.5, APA.\n")

    dummy_journal_file = os.path.join(
        settings.default_journal_path, "dummy_journal_entry.txt"
    )
    if not os.path.exists(dummy_journal_file):
        logger.info("Creating dummy journal file for testing: %s", dummy_journal_file)
        with open(dummy_journal_file, "w", encoding="utf-8") as f:
            f.write("Ceci est une entrée de journal factice pour Jérôme et TP.\n")
            f.write("J'ai travaillé sur le projet Héraclès avec Alexandre Morel.\n")

    graph_app = create_graph(checkpointer_path=":memory:")
    thread_id = str(uuid.uuid4())  # Assurez-vous que uuid est importé
    config = {"configurable": {"thread_id": thread_id}}
    logger.info("Invoking graph with thread_id: %s", thread_id)
    initial_input = AgentState(recreate_vector_store=True)
    final_state_values = None
    try:
        for event_chunk in graph_app.stream(
            initial_input, config=config, stream_mode="values"
        ):
            final_state_values = event_chunk
            node_name = final_state_values.get("last_successful_node", "Unknown")
            logger.info("\n--- State after node: %s ---", node_name)
            for key, value in final_state_values.items():
                if key not in [
                    "school_guidelines_raw_text",
                    "raw_journal_entries",
                ]:
                    logger.info("  %s: %s", key, value)
                elif key == "school_guidelines_raw_text" and value:
                    logger.info("  %s: (%.70s...)", key, value)  # Réduit pour E501
                elif key == "raw_journal_entries" and value:
                    logger.info("  %s: (contains %d entries)", key, len(value))

        if final_state_values:
            logger.info("\n--- Final Graph State Output ---")
            if final_state_values.get("error_message"):
                error_msg = final_state_values["error_message"]
                logger.error("  Error in final state: %s", error_msg)
                if final_state_values.get("error_details"):
                    details = final_state_values["error_details"]
                    logger.error("  Error details: \n%s", details)
        else:
            logger.warning("Graph did not produce a final state output in stream.")
    except Exception as e:
        logger.error("Error running the graph stream: %s", e, exc_info=True)
