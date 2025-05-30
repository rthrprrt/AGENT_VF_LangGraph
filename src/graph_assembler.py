# src/graph_assembler.py
import logging
import os
import uuid

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from src.config import settings
from src.nodes.n0_initial_setup import (
    N0InitialSetupNode,
)

# Correction: Importer la classe
from src.nodes.n1_guideline_ingestor import N1GuidelineIngestorNode  # Correction
from src.nodes.n2_journal_ingestor_anonymizer import (  # Correction
    N2JournalIngestorAnonymizerNode,
)
from src.nodes.n3_thesis_outline_planner import N3ThesisOutlinePlannerNode
from src.nodes.n4_section_processor_router import N4SectionProcessorRouter
from src.nodes.n5_context_retrieval import N5ContextRetrievalNode
from src.nodes.n6_section_drafting import N6SectionDraftingNode
from src.nodes.n8_human_review_hitl_node import N8HumanReviewHITLNode
from src.state import AgentState

logger = logging.getLogger(__name__)


def create_graph(
    checkpointer_path: str | None = None,
):  # Permettre None pour utiliser settings
    """Creates and configures the LangGraph StateGraph for AGENT_VF."""
    logger.info("Creating AGENT_VF_LangGraph workflow...")

    actual_checkpointer_path = checkpointer_path or settings.persistence_db_path
    memory: SqliteSaver

    if actual_checkpointer_path == ":memory:":
        memory = SqliteSaver.from_conn_string(":memory:")
        logger.info("Using in-memory SqliteSaver for persistence.")
    else:
        db_dir = os.path.dirname(actual_checkpointer_path)
        if db_dir:  # pragma: no cover (difficile à tester unitairement sans FS mock)
            os.makedirs(db_dir, exist_ok=True)
        memory = SqliteSaver.from_conn_string(actual_checkpointer_path)
        logger.info("Using SQLiteSaver for persistence: %s", actual_checkpointer_path)

    workflow = StateGraph(AgentState)

    # Instancier les nœuds
    n0_node = N0InitialSetupNode()
    n1_node = N1GuidelineIngestorNode()
    n2_node = N2JournalIngestorAnonymizerNode()
    n3_node = N3ThesisOutlinePlannerNode(llm_model_name=settings.llm_model_name)
    n4_router_node = N4SectionProcessorRouter()
    n5_node = N5ContextRetrievalNode()
    n6_node = N6SectionDraftingNode()  # LLM est initialisé dans son __init__
    n8_node = N8HumanReviewHITLNode()

    # Ajouter les nœuds au graphe en utilisant leurs méthodes `run`
    workflow.add_node("N0_InitialSetupNode", n0_node.run)
    workflow.add_node("N1_GuidelineIngestorNode", n1_node.run)
    workflow.add_node("N2_JournalIngestorAnonymizerNode", n2_node.run)
    workflow.add_node("N3_ThesisOutlinePlannerNode", n3_node.run)
    workflow.add_node("N4_SectionProcessorRouterNode", n4_router_node.run)
    workflow.add_node("N5_ContextRetrievalNode", n5_node.run)
    workflow.add_node("N6_SectionDraftingNode", n6_node.run)
    workflow.add_node("N8_HumanReviewHITLNode", n8_node.run)
    # N7 et les nœuds de compilation/bibliographie seront ajoutés plus tard

    # Définir les points d'entrée et les arêtes
    workflow.set_entry_point("N0_InitialSetupNode")
    workflow.add_edge("N0_InitialSetupNode", "N1_GuidelineIngestorNode")
    workflow.add_edge("N1_GuidelineIngestorNode", "N2_JournalIngestorAnonymizerNode")
    workflow.add_edge("N2_JournalIngestorAnonymizerNode", "N3_ThesisOutlinePlannerNode")
    workflow.add_edge("N3_ThesisOutlinePlannerNode", "N4_SectionProcessorRouterNode")

    # Logique conditionnelle après N4
    workflow.add_conditional_edges(
        "N4_SectionProcessorRouterNode",
        lambda state: state.next_node_override,  # Le routeur met à jour ce champ
        {
            "N5_ContextRetrievalNode": "N5_ContextRetrievalNode",
            "N9_BibliographyManagerNode": END,  # Supposons N9 comme fin pour l'instant
            "ERROR_HANDLER": END,  # Gérer les erreurs en terminant
        },
    )

    workflow.add_edge("N5_ContextRetrievalNode", "N6_SectionDraftingNode")
    # Après N6, on ira vers N7 (Critique) puis N8 (Revue Humaine)
    # Pour l'instant, simplifions en allant vers N8 directement
    workflow.add_edge("N6_SectionDraftingNode", "N8_HumanReviewHITLNode")

    # Après N8, la logique de reprise dépendra de l'état (interrupt ou processed)
    # et sera gérée par le router N4 lors du prochain passage.
    # Si N8 a traité une réponse, il met à jour current_section_index_for_router
    # et le graphe retourne à N4.
    # Si N8 a mis en place un interrupt_payload, le graphe s'arrête et attend.
    # À la reprise, il devrait aussi passer par N4.
    workflow.add_edge("N8_HumanReviewHITLNode", "N4_SectionProcessorRouterNode")

    app = workflow.compile(checkpointer=memory)
    logger.info("AGENT_VF_LangGraph workflow compiled successfully.")
    return app


if __name__ == "__main__":  # pragma: no cover
    # Créer les répertoires et fichiers factices si nécessaire pour l'exécution directe
    os.makedirs(settings.default_output_directory, exist_ok=True)
    if settings.default_school_guidelines_path:
        guidelines_dir = os.path.dirname(settings.default_school_guidelines_path)
        if guidelines_dir:
            os.makedirs(guidelines_dir, exist_ok=True)
        if not os.path.exists(settings.default_school_guidelines_path):
            logger.info(
                "Creating dummy guideline file for testing: %s",
                settings.default_school_guidelines_path,
            )
            with open(
                settings.default_school_guidelines_path, "w", encoding="utf-8"
            ) as f:
                f.write("Ceci est un document de directives factice.\n")
                f.write("Section: Introduction\nContenu de l'intro.\n")
                f.write("Format: Times New Roman 12, interligne 1.5, APA.\n")

    if settings.default_journal_path:
        journal_dir = settings.default_journal_path
        os.makedirs(journal_dir, exist_ok=True)
        dummy_journal_file = os.path.join(journal_dir, "dummy_journal_entry.txt")
        if not os.path.exists(dummy_journal_file):
            logger.info(
                "Creating dummy journal file for testing: %s", dummy_journal_file
            )
            with open(dummy_journal_file, "w", encoding="utf-8") as f:
                f.write("Ceci est une entrée de journal factice pour Jérôme et TP.\n")
                f.write("J'ai travaillé sur le projet Héraclès avec Alexandre Morel.\n")

    graph_app = create_graph()  # Utilise le checkpointer par défaut depuis settings
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    logger.info("Invoking graph with thread_id: %s", thread_id)

    # Entrée initiale minimale, N0 complétera
    initial_input_dict = {"recreate_vector_store": True}
    # Convertir en AgentState si nécessaire pour la méthode stream,
    # ou passer le dict directement si StateGraph le gère.
    # LangGraph v0.1+ préfère un dict pour l'input initial d'un stream.

    final_state_output: dict | None = None
    try:
        for event_chunk in graph_app.stream(
            initial_input_dict, config=config, stream_mode="values"
        ):
            final_state_output = (
                event_chunk  # Garder la dernière version complète de l'état
            )
            # Afficher les informations clés après chaque nœud
            node_name = final_state_output.get("last_successful_node", "Unknown Node")
            current_op = final_state_output.get(
                "current_operation_message", "No message"
            )
            error_msg_state = final_state_output.get("error_message")

            logger.info("\n--- State after node: %s ---", node_name)
            logger.info("  Current Operation: %s", current_op)
            if error_msg_state:
                logger.error("  Error Message in State: %s", error_msg_state)

            # Afficher l'ID de la section courante si disponible
            current_section_id = final_state_output.get("current_section_id")
            if current_section_id:
                logger.info("  Current Section ID: %s", current_section_id)

        if final_state_output:
            logger.info("\n--- Final Graph State Output (last streamed value) ---")
            if final_state_output.get("error_message"):
                logger.error(
                    "  Error in final state: %s", final_state_output["error_message"]
                )
                if final_state_output.get("error_details"):
                    logger.error(
                        "  Error details: \n%s", final_state_output["error_details"]
                    )
            # Afficher le plan généré (titres des sections)
            if final_state_output.get("thesis_outline"):
                logger.info("  Generated Thesis Outline:")
                for section_data in final_state_output["thesis_outline"]:
                    # section_data est un dict ici car AgentState a été converti
                    logger.info(
                        "    - ID: %s, Title: %s, Status: %s",
                        section_data.get("id"),
                        section_data.get("title"),
                        section_data.get("status"),
                    )
        else:  # pragma: no cover
            logger.warning("Graph did not produce a final state output in stream.")

    except Exception as e:  # pragma: no cover
        logger.error("Error running the graph stream: %s", e, exc_info=True)
