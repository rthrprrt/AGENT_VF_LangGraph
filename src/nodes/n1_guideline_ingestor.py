# src/nodes/n1_guideline_ingestor.py
import logging
import re
from typing import Any

import pypdf

from src.state import AgentState

logger = logging.getLogger(__name__)

FONT_PATTERN = re.compile(
    r"police.*?(Times New Roman|Arial|Calibri)[\s,]*(\d+)\s*pts", re.IGNORECASE
)
LINE_SPACING_PATTERN = re.compile(
    r"interligne.*?(simple|1\.5|double|personnalisé(?:[\s:]*([\d,.]+)\s*pts)?)",
    re.IGNORECASE,
)
CITATION_STYLE_PATTERN = re.compile(
    r"style\s+de\s+citation.*?(APA|Harvard|MLA|Chicago)", re.IGNORECASE
)
MIN_PAGES_PATTERN = re.compile(r"minimum\sde\s*(\d+)\s*pages", re.IGNORECASE)
TEXT_JUSTIFICATION_PATTERN = re.compile(r"texte\s+justifié", re.IGNORECASE)

# Ce pattern simple a fonctionné pour les tests de base.
# Il capture un chiffre suivi d'un point, puis des espaces, puis le reste de la ligne comme titre.
SECTION_TITLE_PATTERN_FOR_TEST = re.compile(
    r"^\s*\d+\.\s*(.+)$", re.MULTILINE | re.IGNORECASE
)


class N1GuidelineIngestorNode:
    """
    Nœud responsable de l'ingestion et du parsing des directives scolaires
    (format PDF) pour en extraire la structure, les contraintes de formatage,
    et les objectifs clés.
    """

    def _extract_text_from_pdf(self, pdf_path: str) -> str | None:
        try:
            reader = pypdf.PdfReader(pdf_path)
            text = ""
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text() or ""
            return text
        except FileNotFoundError:
            logger.error("Guidelines PDF file not found at %s", pdf_path)
            return None
        except Exception as e:  # noqa: BLE001
            logger.error(
                "Failed to read or parse PDF file at %s: %s", pdf_path, e, exc_info=True
            )
            return None

    def _parse_formatting_rules(self, text: str) -> dict[str, Any]:
        rules: dict[str, Any] = {}
        font_match = FONT_PATTERN.search(text)
        if font_match:
            rules["font_name"] = font_match.group(1).strip()
            rules["font_size"] = int(font_match.group(2))

        spacing_match = LINE_SPACING_PATTERN.search(text)
        if spacing_match:
            spacing_type_raw = spacing_match.group(1).lower().strip()
            # Le groupe 2 est optionnel et ne sera présent que pour "personnalisé XX pts"
            if "personnalisé" in spacing_type_raw and spacing_match.group(2):
                rules["line_spacing_type"] = "personnalisé"
                try:
                    rules["line_spacing_pts"] = float(
                        spacing_match.group(2).replace(",", ".")
                    )
                except (
                    ValueError,
                    AttributeError,
                ):  # Attraper si group(2) est None ou non convertible
                    logger.warning(
                        f"Could not parse line spacing points from: {spacing_match.group(0)}"
                    )
            elif spacing_type_raw in ["1.5", "simple", "double"]:
                rules["line_spacing_type"] = spacing_type_raw

        citation_match = CITATION_STYLE_PATTERN.search(text)
        if citation_match:
            rules["citation_style"] = citation_match.group(1).upper()

        pages_match = MIN_PAGES_PATTERN.search(text)
        if pages_match:
            rules["min_pages"] = int(pages_match.group(1))

        rules["text_justification"] = bool(TEXT_JUSTIFICATION_PATTERN.search(text))
        return rules

    def _parse_structured_content(self, text: str) -> dict[str, list[str]]:
        structured_content: dict[str, list[str]] = {}
        # Utiliser le pattern qui fonctionnait pour les tests initiaux
        for match in SECTION_TITLE_PATTERN_FOR_TEST.finditer(text):
            title = match.group(1).strip().replace("\n", " ")
            # Correction de la condition:
            # Accepter les titres d'au moins 4 caractères, même sans espace (ex: "INTRODUCTION")
            # et qui ne commencent pas par "page" (pour éviter de capturer "Page X sur Y")
            # Le pattern capture déjà ce qui suit "X. ", donc "1. Contexte" (sous-section) n'est pas ce que l'on vise ici.
            # Pour l'instant, ce pattern simple ne distingue pas les niveaux de sous-section.
            # Une amélioration future pourrait utiliser des patterns plus spécifiques par niveau.
            # "1. Introduction." -> title = "Introduction." -> "Introduction"
            # "1. INTRODUCTION" -> title = "INTRODUCTION"
            # "2. Description de la mission." -> title = "Description de la mission." -> "Description de la mission"
            # "VI. CONCLUSION" ne sera pas capturé par ce pattern.

            # Pour capturer "1. Introduction" ou "2. Titre avec espaces" mais pas "1.1. Sous-titre"
            # le pattern pourrait être r"^\s*(\d+)\.\s+([A-ZÀ-Ÿ][A-ZÀ-Ÿ\s\-']{3,})$"
            # Mais le test plan dit "Ce pattern simple a fonctionné pour les tests de base."
            # Je vais donc juste modifier la condition de filtrage du titre.

            cleaned_title = title.rstrip(
                "."
            )  # Enlever le point final si présent pour la clé

            if len(cleaned_title) >= 4 and not cleaned_title.lower().startswith("page"):
                # S'assurer que le titre ne contient pas lui-même une structure de sous-section
                # comme "1. Contexte" (ce qui indique que le pattern a capturé trop loin)
                if not re.match(r"^\d+\.\d+", cleaned_title):  # Eviter X.Y ...
                    structured_content[cleaned_title] = []
        return structured_content

    def run(self, state: AgentState) -> dict[str, Any]:
        logger.info(
            "N1: Ingesting school guidelines from: %s...", state.school_guidelines_path
        )
        updated_fields: dict[str, Any] = {}

        if not state.school_guidelines_path:
            msg = "N1: School guidelines path is not set."
            logger.error(msg)
            updated_fields["error_message"] = msg
            updated_fields["last_successful_node"] = "N1GuidelineIngestorNode_Error"
            return updated_fields

        raw_text = self._extract_text_from_pdf(state.school_guidelines_path)

        if raw_text is None:
            msg = f"N1: Failed to extract text from PDF: {state.school_guidelines_path}"
            logger.error(msg)  # Déjà loggé par _extract_text_from_pdf
            updated_fields["error_message"] = msg
            updated_fields["last_successful_node"] = "N1GuidelineIngestorNode_Error"
            return updated_fields

        logger.info(
            "  Successfully read %d characters from guidelines PDF.", len(raw_text)
        )
        updated_fields["school_guidelines_raw_text"] = raw_text

        formatting_rules = self._parse_formatting_rules(raw_text)
        updated_fields["school_guidelines_formatting"] = formatting_rules
        logger.info("  Parsed formatting rules: %s", formatting_rules)

        structured_content = self._parse_structured_content(raw_text)
        updated_fields["school_guidelines_structured"] = structured_content
        logger.info(
            "  Parsed structured content (sections found): %s",
            list(structured_content.keys()),
        )

        if not structured_content and not formatting_rules and len(raw_text) > 0:
            logger.warning(
                "N1: No structured content or formatting rules were parsed from non-empty PDF. "
                "Patterns may need adjustment or PDF content is not as expected."
            )
        elif not structured_content and len(raw_text) > 0:
            logger.warning(
                "N1: No structured content (sections) parsed from non-empty PDF."
            )
        elif not formatting_rules and len(raw_text) > 0:
            logger.warning("N1: No formatting rules parsed from non-empty PDF.")

        updated_fields["last_successful_node"] = "N1GuidelineIngestorNode"
        logger.info("N1: Guideline ingestion complete.")
        return updated_fields
