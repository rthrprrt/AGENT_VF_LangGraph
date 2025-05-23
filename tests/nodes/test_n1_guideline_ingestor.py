# tests/nodes/test_n1_guideline_ingestor.py
# Standard library import (os n'était pas utilisé, supprimé)
from unittest.mock import MagicMock, patch

# from pypdf import PdfReader # Pas besoin ici car mocké

from src.nodes.n1_guideline_ingestor import (
    _parse_formatting_rules,
    _parse_structured_content,
    ingest_school_guidelines,
)
from src.state import AgentState


MOCK_PDF_CONTENT_PAGE_1 = """
Mission Professionnelle Digi5
Format et Présentation
Longueur: Le mémoire doit contenir au minimum 30 pages hors annexes.
Mise en page : Utilisez une police claire et adaptée (Times New Roman 12) """ \
"""avec un interligne de 1.5. Justifiez le texte.
Citations et bibliographie: Suivez un style de citation cohérent """ \
"""(APA ou Harvard).

1. Introduction
Décrivez l'entreprise...
"""

MOCK_PDF_CONTENT_PAGE_2 = """
2. Description de la mission
Expliquez votre fiche de poste...

3. Conclusion
Synthèse des apprentissages...
"""


def test_parse_formatting_rules():
    """Tests extraction of formatting rules."""
    rules = _parse_formatting_rules(MOCK_PDF_CONTENT_PAGE_1)
    assert rules.get("font_name") == "Times New Roman"
    assert rules.get("font_size") == 12
    assert rules.get("line_spacing") == 1.5
    assert rules.get("citation_style") == "APA"
    assert rules.get("min_pages") == 30
    assert rules.get("text_justification") is True


def test_parse_structured_content():
    """Tests extraction of structured content (sections)."""
    full_mock_text = MOCK_PDF_CONTENT_PAGE_1 + MOCK_PDF_CONTENT_PAGE_2
    structure = _parse_structured_content(full_mock_text)
    assert "Introduction" in structure
    assert "Description de la mission" in structure
    assert "Conclusion" in structure
    assert "Décrivez l'entreprise..." in structure["Introduction"]
    assert "Expliquez votre fiche de poste..." in structure[
        "Description de la mission"
    ]


def test_ingest_school_guidelines_success():
    """Tests successful ingestion and parsing of guidelines."""
    initial_state = AgentState(school_guidelines_path="dummy/path.pdf")

    mock_page1 = MagicMock()
    mock_page1.extract_text.return_value = MOCK_PDF_CONTENT_PAGE_1
    mock_page2 = MagicMock()
    mock_page2.extract_text.return_value = MOCK_PDF_CONTENT_PAGE_2

    mock_pdf_reader_instance = MagicMock()
    mock_pdf_reader_instance.pages = [mock_page1, mock_page2]

    with patch(
        "src.nodes.n1_guideline_ingestor.PdfReader",
        return_value=mock_pdf_reader_instance,
    ) as mock_pdf_reader_class:
        updated_state = ingest_school_guidelines(initial_state)

        mock_pdf_reader_class.assert_called_once_with("dummy/path.pdf")
        assert updated_state.error_message is None # Doit être None si succès
        assert MOCK_PDF_CONTENT_PAGE_1.strip() in (
            updated_state.school_guidelines_raw_text or ""
        )
        assert MOCK_PDF_CONTENT_PAGE_2.strip() in (
            updated_state.school_guidelines_raw_text or ""
        )
        assert updated_state.school_guidelines_formatting is not None
        assert updated_state.school_guidelines_formatting.get(
            "font_name"
        ) == "Times New Roman"
        assert "Introduction" in (
            updated_state.school_guidelines_structured or {}
        )
        assert updated_state.last_successful_node == "N1_GuidelineIngestorNode"

def test_ingest_school_guidelines_file_not_found():
    """Tests handling of FileNotFoundError."""
    initial_state = AgentState(school_guidelines_path="non_existent_file.pdf")
    with patch(
        "src.nodes.n1_guideline_ingestor.PdfReader",
        side_effect=FileNotFoundError("File not found"), # Simule l'erreur exacte
    ) as mock_pdf_reader_class:
        updated_state = ingest_school_guidelines(initial_state)
        mock_pdf_reader_class.assert_called_once_with("non_existent_file.pdf")
        # Vérifie que le message d'erreur contient la partie attendue
        expected_error_part = "Guidelines PDF file not found at non_existent_file.pdf"
        assert expected_error_part in (updated_state.error_message or "")


def test_ingest_school_guidelines_no_text_extracted():
    """Tests handling when PDF yields no text."""
    initial_state = AgentState(school_guidelines_path="dummy/empty.pdf")
    mock_page_empty = MagicMock()
    mock_page_empty.extract_text.return_value = ""
    mock_pdf_reader_instance = MagicMock()
    mock_pdf_reader_instance.pages = [mock_page_empty]

    with patch(
        "src.nodes.n1_guideline_ingestor.PdfReader",
        return_value=mock_pdf_reader_instance
    ):
        updated_state = ingest_school_guidelines(initial_state)
        assert "No text content extracted" in (updated_state.error_message or "")
        assert updated_state.school_guidelines_raw_text == ""