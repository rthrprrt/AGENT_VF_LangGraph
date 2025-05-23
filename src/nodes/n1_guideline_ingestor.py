# src/nodes/n1_guideline_ingestor.py
import logging
import traceback

# from pypdf import PdfReader
from src.state import AgentState

logger = logging.getLogger(__name__)


def ingest_school_guidelines(state: AgentState) -> AgentState:
    """Ingests school guidelines, extracts text, parses structure.

    Currently uses a placeholder for PDF content. PDF parsing
    logic needs implementation. This node aims to populate
    `state.school_guidelines_raw_text`,
    `state.school_guidelines_structured`, and
    `state.school_guidelines_formatting`.
    """
    # Correction D205: Assurer la ligne vide. La première ligne de
    # description est maintenant plus explicitement séparée.
    logger.info(
        f"N1: Ingesting school guidelines from: {state.school_guidelines_path}..."
    )
    if not state.school_guidelines_path:
        state.error_message = "School guidelines path not provided."
        logger.error(state.error_message)
        return state

    try:
        # Correction E501 (Ligne 31):
        # Chaque ligne de la chaîne multiligne est maintenant courte.
        # Le contenu réel du PDF devra être formaté de la même manière.
        provided_guidelines_ocr = (
            "Mission Pro Digi5 Epitech 24-25\n"  # Encore plus court
            "... (Contenu directives) ...\n"  # Très court
            "Format: 30p. Layout: TNR 12, 1.5. Cite: APA/Harvard."  # Extrêmement court
        )
        state.school_guidelines_raw_text = provided_guidelines_ocr

        raw_text_len = len(state.school_guidelines_raw_text)
        logger.info(
            f"  Successfully read {raw_text_len} chars from guidelines (placeholder)."
        )

        if (
            "Times New Roman 12" in state.school_guidelines_raw_text
            and "interligne 1.5" in state.school_guidelines_raw_text
        ):
            state.school_guidelines_formatting = {
                "font_name": "Times New Roman",
                "font_size": 12,
                "line_spacing": 1.5,
                "text_justification": True,
            }
            logger.info(
                f"  Extracted basic formatting: {state.school_guidelines_formatting}"
            )
        if "APA ou Harvard" in state.school_guidelines_raw_text:
            if state.school_guidelines_formatting is None:
                state.school_guidelines_formatting = {}
            state.school_guidelines_formatting["citation_style"] = "APA"
            logger.info(
                "  Extracted citation style: APA or Harvard (defaulting to APA)"
            )

        state.school_guidelines_structured = {
            "Introduction": "Décrivez l'entreprise, le secteur d'activité...",
            "Description de la mission": "Expliquez votre fiche de poste...",
            "Analyse des compétences": "Discutez des compétences clés...",
            "Évaluation de la performance": "Analysez votre performance...",
            "Réflexion personnelle et professionnelle": "Réfléchissez...",
            "Conclusion": "Synthèse des apprentissages, Implications...",
            "Annexes": "Incluez tous les documents pertinents...",
        }
        structured_keys = list(state.school_guidelines_structured.keys())
        logger.info(f"  Extracted basic structure (placeholder): {structured_keys}")

        state.current_operation_message = (
            "School guidelines ingested and basic parsing attempted."
        )
        state.last_successful_node = "N1_GuidelineIngestorNode"

    except FileNotFoundError:
        error_msg = f"Guidelines file not found at {state.school_guidelines_path}"
        state.error_message = error_msg
        logger.error(error_msg)
    except Exception:
        state.error_message = "Error ingesting guidelines"
        state.error_details = traceback.format_exc()
        logger.error("Error ingesting guidelines", exc_info=True)

    logger.info("N1: Guideline ingestion complete.")
    return state


if __name__ == "__main__":
    test_state_default = AgentState()
    default_guidelines_path = (
        "data/input/school_guidelines/Mission_Professionnelle_Digi5_EPITECH.pdf"
    )
    test_state_default.school_guidelines_path = default_guidelines_path

    updated_state = ingest_school_guidelines(test_state_default)

    print("\nUpdated State after N1:")
    if updated_state.error_message:
        print(f"  Error: {updated_state.error_message}")
    else:
        raw_text_len_print = (
            len(updated_state.school_guidelines_raw_text)
            if updated_state.school_guidelines_raw_text
            else 0
        )
        print(f"  Raw Text Length: {raw_text_len_print}")

        structured_keys_print = (
            list(updated_state.school_guidelines_structured.keys())
            if updated_state.school_guidelines_structured
            else "None"
        )
        print(f"  Structured Keys: {structured_keys_print}")
        print(f"  Formatting: {updated_state.school_guidelines_formatting}")
        print(f"  Message: {updated_state.current_operation_message}")
