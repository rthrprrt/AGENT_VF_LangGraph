# src/nodes/n1_guideline_ingestor.py
import logging
import re
import traceback
from typing import Any

from pypdf import PdfReader

from src.state import AgentState

logger = logging.getLogger(__name__)

SECTION_KEYWORDS = [
    "Introduction",
    "Description de la mission",
    "Analyse des compétences",
    "Évaluation de la performance",
    "Réflexion personnelle et professionnelle",
    "Conclusion",
    "Annexes",
]

FORMATTING_PATTERNS = {
    "font_name_size": r"(Times New Roman)\s*(\d+)",
    "line_spacing": r"interligne de\s*([\d]+\.[\d]+|[\d]+)\s*\.?",
    "citation_style": (r"style de citation cohérent\s*\((APA ou Harvard)\)"),
    "min_pages": r"minimum\s*(\d+)\s*pages",
}


def _parse_formatting_rules(text: str) -> dict[str, Any]:
    """Extracts formatting rules from text using regex patterns."""
    rules: dict[str, Any] = {}
    font_match = re.search(FORMATTING_PATTERNS["font_name_size"], text, re.IGNORECASE)
    if font_match:
        rules["font_name"] = font_match.group(1)
        rules["font_size"] = int(font_match.group(2))

    spacing_match = re.search(FORMATTING_PATTERNS["line_spacing"], text, re.IGNORECASE)
    if spacing_match:
        captured_spacing = spacing_match.group(1)
        try:
            rules["line_spacing"] = float(captured_spacing)
        except ValueError as e:
            logger.warning(
                "Could not convert spacing value '%s' to float: %s", captured_spacing, e
            )

    citation_match = re.search(
        FORMATTING_PATTERNS["citation_style"], text, re.IGNORECASE
    )
    if citation_match:
        style_components = citation_match.group(1).split(" ou ")
        rules["citation_style"] = style_components[0].strip()

    pages_match = re.search(FORMATTING_PATTERNS["min_pages"], text, re.IGNORECASE)
    if pages_match:
        rules["min_pages"] = int(pages_match.group(1))

    if "justifi" in text.lower():
        rules["text_justification"] = True
    return rules


def _add_section_if_exists(
    structure: dict[str, str], title: str | None, content: list[str]
):
    """Helper: adds content to structure if title and content exist."""
    if title and content:
        structure[title] = "\n".join(content).strip()


def _find_section_keyword_in_line(line: str) -> str | None:
    """Helper: finds a section keyword in a line."""
    if len(line.strip()) < 50:
        for keyword in SECTION_KEYWORDS:
            if keyword.lower() in line.lower():
                return keyword
    return None


def _parse_structured_content(text: str) -> dict[str, str]:
    """
    Extracts thesis sections and their high-level requirements.

    Refactored to reduce complexity.
    """
    structure: dict[str, str] = {}
    current_section_content: list[str] = []
    current_section_title: str | None = None

    lines = text.splitlines()
    for line in lines:
        stripped_line = line.strip()
        found_keyword = _find_section_keyword_in_line(stripped_line)

        if found_keyword:
            _add_section_if_exists(
                structure, current_section_title, current_section_content
            )
            current_section_title = found_keyword
            # Ligne 120 (potentiellement E501, coupée si nécessaire)
            current_section_content = [stripped_line]  # Inclut la ligne du titre
        elif current_section_title:
            current_section_content.append(line)

    _add_section_if_exists(structure, current_section_title, current_section_content)

    if not structure and text:
        logger.warning(
            "Could not parse detailed structure from guidelines, "
            "using fallback 'ContenuGénéral'."
        )
        structure["ContenuGénéral"] = text
    elif not structure:
        logger.warning("No text found in guidelines to parse structure from.")
    return structure


def ingest_school_guidelines(state: AgentState) -> AgentState:
    """Ingests PDF guidelines, extracts text, and parses structure/formatting."""
    logger.info(
        "N1: Ingesting school guidelines from: %s...", state.school_guidelines_path
    )
    if not state.school_guidelines_path:
        state.error_message = "School guidelines path not provided in state."
        logger.error(state.error_message)
        return state

    raw_text = ""
    try:
        reader = PdfReader(state.school_guidelines_path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                raw_text += page_text + "\n"
        state.school_guidelines_raw_text = raw_text.strip()
        logger.info(
            "  Successfully read %d characters from guidelines PDF.", len(raw_text)
        )

        if not raw_text.strip():
            logger.warning("  No text content extracted from the PDF.")
            state.error_message = "No text content extracted from PDF guidelines."
            return state

        state.school_guidelines_formatting = _parse_formatting_rules(raw_text)
        logger.info("  Parsed formatting rules: %s", state.school_guidelines_formatting)

        state.school_guidelines_structured = _parse_structured_content(raw_text)
        keys_found = []
        if state.school_guidelines_structured:
            keys_found = list(state.school_guidelines_structured.keys())
        # Ligne 163 (potentiellement E501, log sur plusieurs lignes)
        logger.info("  Parsed structured content (sections found): %s", keys_found)

        state.current_operation_message = "School guidelines ingested and parsed."
        state.last_successful_node = "N1_GuidelineIngestorNode"

    except FileNotFoundError:
        state.error_message = (
            f"Guidelines PDF file not found at {state.school_guidelines_path}"
        )
        logger.error(state.error_message)
    except Exception as e:
        state.error_message = f"Error during guidelines ingestion: {str(e)}"
        state.error_details = traceback.format_exc()
        logger.error("Error during guidelines ingestion: %s", e, exc_info=True)

    logger.info("N1: Guideline ingestion complete.")
    return state
