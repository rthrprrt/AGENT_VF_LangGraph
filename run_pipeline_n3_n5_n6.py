import logging
import json
from pathlib import Path

from src.config import settings
from src.state import AgentState, SectionDetail
from src.nodes.n0_initial_setup import N0InitialSetupNode
from src.nodes.n1_guideline_ingestor import N1GuidelineIngestorNode
from src.nodes.n2_journal_ingestor_anonymizer import N2JournalIngestorAnonymizerNode # IMPORT AJOUTÉ
from src.nodes.n3_thesis_outline_planner import N3ThesisOutlinePlannerNode
from src.nodes.n5_context_retrieval import N5ContextRetrievalNode
from src.nodes.n6_section_drafting import N6SectionDraftingNode

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("Début du pipeline de test N0 -> N1 -> N2 -> N3 -> N5 -> N6")

    school_guidelines_pdf_path = settings.default_school_guidelines_path
    example_thesis_file_path = "data/input/school_guidelines/Mémoire de Mission Professionnelle – Digi5.txt"

    try:
        with open(example_thesis_file_path, "r", encoding="utf-8") as f:
            digi5_content = f.read()
        logger.info(f"Contenu de '{example_thesis_file_path}' chargé.")
    except FileNotFoundError:
        logger.error(f"ERREUR: Fichier exemple de thèse '{example_thesis_file_path}' non trouvé.")
        # ... (messages d'erreur existants) ...
        return
    except Exception as e:
        logger.error(f"Erreur lors du chargement de '{example_thesis_file_path}': {e}")
        return

    initial_state_dict = {
        "school_guidelines_path": school_guidelines_pdf_path,
        "journal_path": settings.default_journal_path,
        "output_directory": "outputs/pipeline_test",
        "vector_store_path": settings.vector_store_directory,
        "llm_model_name": settings.llm_model_name,
        "embedding_model_name": settings.embedding_model_name,
        # Mettre à True pour la première exécution pour créer/recréer le vector store
        "recreate_vector_store": True, 
        "user_persona": (
            "un(e) étudiant(e) en dernière année de Master spécialisé en IA et "
            "transformation d'entreprise, réalisant son alternance en tant que "
            "chef de projet IA (AIPO) dans une foncière immobilière."
        ),
        "example_thesis_text_content": digi5_content,
        "max_reflection_attempts": 3,
    }
    current_state = AgentState(**initial_state_dict)
    logger.info("État initial préparé.")

    # --- Étape N0: Initial Setup ---
    logger.info("\n--- EXÉCUTION N0: InitialSetupNode ---")
    n0_node = N0InitialSetupNode()
    n0_output = n0_node.run(current_state)
    current_state = AgentState(**{**current_state.dict(), **n0_output})
    if current_state.error_message:
        logger.error(f"Erreur N0: {current_state.error_message}")
        return
    logger.info(f"N0 terminé. Message: {current_state.current_operation_message}") # Message peut être None
    logger.info(f"  Chemin Vector Store utilisé: {current_state.vector_store_path}")


    # --- Étape N1: Guideline Ingestor ---
    logger.info("\n--- EXÉCUTION N1: GuidelineIngestorNode ---")
    n1_node = N1GuidelineIngestorNode()
    n1_output = n1_node.run(current_state)
    current_state = AgentState(**{**current_state.dict(), **n1_output})
    if current_state.error_message:
        logger.error(f"Erreur N1: {current_state.error_message}")
        return
    logger.info(f"N1 terminé. Message: {current_state.current_operation_message}")
    if current_state.school_guidelines_structured:
        # Afficher seulement les clés pour la concision
        logger.info(f"  Directives structurées (clés): {list(current_state.school_guidelines_structured.keys())}")
    else:
        logger.warning("  Aucune directive structurée n'a été parsée par N1.")

    # --- ÉTAPE N2: Journal Ingestor & Anonymizer (CRÉATION DU VECTOR STORE) ---
    logger.info("\n--- EXÉCUTION N2: JournalIngestorAnonymizerNode ---")
    n2_node = N2JournalIngestorAnonymizerNode() # Utilise les paramètres par défaut pour chunk_size/overlap
    n2_output = n2_node.run(current_state)
    current_state = AgentState(**{**current_state.dict(), **n2_output})
    if current_state.error_message:
        logger.error(f"Erreur N2: {current_state.error_message}")
        return
    logger.info(f"N2 terminé. Message: {current_state.current_operation_message}")
    logger.info(f"  Vector store initialisé: {current_state.vector_store_initialized}")


    # --- Étape N3: Thesis Outline Planner ---
    logger.info("\n--- EXÉCUTION N3: ThesisOutlinePlannerNode (avec LLM réel) ---")
    n3_node = N3ThesisOutlinePlannerNode(
        llm_model_name=current_state.llm_model_name or settings.llm_model_name
    )
    n3_output = n3_node.run(current_state)
    current_state = AgentState(**{**current_state.dict(), **n3_output})

    if current_state.error_message:
        logger.error(f"Erreur N3: {current_state.error_message}")
        if current_state.error_details:
            logger.error(f"Détails Erreur N3: {current_state.error_details}")
        return
    logger.info(f"N3 terminé. Message: {current_state.current_operation_message}")
    
    if not current_state.thesis_outline:
        logger.error("N3 n'a généré aucun plan (thesis_outline est vide). Arrêt.")
        return
        
    logger.info("Plan de thèse généré (" + str(len(current_state.thesis_outline)) + " sections):")
    for i, section_detail_obj in enumerate(current_state.thesis_outline):
        logger.info(f"  {i+1}. ID: {section_detail_obj.id}, Titre: {section_detail_obj.title}")
        # logger.info(f"     Keywords pour N5: {section_detail_obj.student_experience_keywords}") # Log un peu verbeux

    # --- Choix de la section à tester ---
    section_to_test_index = -1
    for i, section_obj in enumerate(current_state.thesis_outline):
        if not section_obj.id.startswith("0.") and section_obj.student_experience_keywords:
            section_to_test_index = i
            break
    
    if section_to_test_index == -1:
        for i, section_obj in enumerate(current_state.thesis_outline):
            if not section_obj.id.startswith("0."):
                section_to_test_index = i
                break
    
    if section_to_test_index == -1 and len(current_state.thesis_outline) > 0:
        section_to_test_index = 0
    elif section_to_test_index == -1 :
        logger.error("Aucune section trouvée dans le plan pour tester N5/N6.")
        return

    chosen_section_detail: SectionDetail = current_state.thesis_outline[section_to_test_index]
    current_state.current_section_id = chosen_section_detail.id
    current_state.current_section_index = section_to_test_index
    logger.info(f"\nSection choisie pour N5/N6: ID={chosen_section_detail.id}, Titre='{chosen_section_detail.title}'")
    logger.info(f"Keywords pour cette section: {chosen_section_detail.student_experience_keywords}")


    # --- Étape N5: Context Retrieval ---
    logger.info("\n--- EXÉCUTION N5: ContextRetrievalNode (avec RAG réel) ---")
    n5_node = N5ContextRetrievalNode()
    n5_output = n5_node.run(current_state)
    current_state = AgentState(**{**current_state.dict(), **n5_output})

    if current_state.error_message:
        logger.error(f"Erreur N5: {current_state.error_message}")
        return
    logger.info(f"N5 terminé. Message: {current_state.current_operation_message}")

    updated_section_after_n5 = current_state.get_section_by_id(chosen_section_detail.id)
    if updated_section_after_n5 and updated_section_after_n5.anonymized_context_for_llm:
        logger.info("Contexte récupéré par N5 pour la section:")
        logger.info("vvv --- Début Contexte N5 --- vvv")
        logger.info(updated_section_after_n5.anonymized_context_for_llm[:1000] + "...") # Tronquer si trop long
        logger.info("^^^ --- Fin Contexte N5 --- ^^^")
    elif updated_section_after_n5:
        logger.warning("Aucun contexte récupéré par N5 pour cette section.")
    else: # pragma: no cover
        logger.error("La section testée n'a pas été retrouvée après N5.")
        return

    # --- Étape N6: Section Drafting ---
    logger.info("\n--- EXÉCUTION N6: SectionDraftingNode (avec LLM réel) ---")
    n6_node = N6SectionDraftingNode()
    n6_output = n6_node.run(current_state)
    current_state = AgentState(**{**current_state.dict(), **n6_output})

    if current_state.error_message:
        logger.error(f"Erreur N6: {current_state.error_message}")
        if current_state.error_details_n6_drafting:
             logger.error(f"Détails Erreur N6: {current_state.error_details_n6_drafting}")
        return
    logger.info(f"N6 terminé. Message: {current_state.current_operation_message}")

    final_section_state = current_state.get_section_by_id(chosen_section_detail.id)
    if final_section_state and final_section_state.draft_v1:
        logger.info(f"\nBrouillon (draft_v1) généré par N6 pour la section '{final_section_state.title}':")
        logger.info("vvv --- Début Brouillon N6 --- vvv")
        logger.info(final_section_state.draft_v1)
        logger.info("^^^ --- Fin Brouillon N6 --- ^^^")
    elif final_section_state: # pragma: no cover
        logger.warning("Aucun draft_v1 trouvé pour la section après N6.")
    else: # pragma: no cover
        logger.error("La section testée n'a pas été retrouvée après N6.")

    logger.info("\nPipeline de test N0 -> N1 -> N2 -> N3 -> N5 -> N6 terminé.")


if __name__ == "__main__":
    output_dir_for_script = Path("outputs/pipeline_test")
    output_dir_for_script.mkdir(parents=True, exist_ok=True)
    
    if settings.default_school_guidelines_path:
        guidelines_file = Path(settings.default_school_guidelines_path)
        if not guidelines_file.exists(): # pragma: no cover
            logger.warning(f"Fichier de directives par défaut non trouvé: {guidelines_file}")
            logger.warning("N1 pourrait échouer ou utiliser un chemin incorrect.")
        else:
            guidelines_file.parent.mkdir(parents=True, exist_ok=True)

    if settings.default_journal_path:
        Path(settings.default_journal_path).mkdir(parents=True, exist_ok=True)
        # Créer un fichier journal factice s'il n'y en a pas, pour que N2 ait quelque chose à traiter
        dummy_journal_entry = Path(settings.default_journal_path) / "2024-01-01_dummy_entry.txt"
        if not dummy_journal_entry.exists(): # pragma: no cover
            logger.info(f"Création d'une entrée de journal factice: {dummy_journal_entry}")
            dummy_journal_entry.write_text("Ceci est une entrée de journal factice pour les tests du pipeline. "
                                           "J'ai travaillé sur le projet Alpha et le projet Automatisation DG. "
                                           "J'ai utilisé Power Automate et AI Builder.")

    main()