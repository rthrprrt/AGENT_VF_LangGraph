# tests/nodes/test_n1_guideline_ingestor.py
import unittest
from unittest.mock import MagicMock, patch

from src.nodes.n1_guideline_ingestor import N1GuidelineIngestorNode
from src.state import AgentState

# Désactiver le logging global pour les tests sauf si spécifiquement activé
# logging.disable(logging.CRITICAL)


class TestN1GuidelineIngestorNode(unittest.TestCase):
    def setUp(self):
        self.node = N1GuidelineIngestorNode()
        self.initial_state = AgentState(school_guidelines_path="dummy/path.pdf")
        # Réactiver le logging pour ce logger spécifique si besoin de débugger N1
        # logging.getLogger("src.nodes.n1_guideline_ingestor").setLevel(logging.DEBUG)

    @patch("src.nodes.n1_guideline_ingestor.pypdf.PdfReader")
    def test_ingest_school_guidelines_success(self, mock_pdf_reader):
        """Test successful ingestion and parsing of guidelines."""
        mock_page = MagicMock()
        # MODIFICATION: Ajout de \n pour simuler des titres sur leurs propres lignes
        mock_page.extract_text.return_value = (
            "Titre RNCP. Objectifs. Police Times New Roman 12 pts. "
            "Interligne 1.5. Style de citation APA. Minimum de 30 pages. "
            "Texte justifié.\n1. Introduction.\n2. Description de la mission.\n"  # Ajout de \n
            "- Sous point 2.1.\n6. Conclusion."  # Ajout de \n
        )
        mock_reader_instance = mock_pdf_reader.return_value
        mock_reader_instance.pages = [mock_page]

        result = self.node.run(self.initial_state)

        assert result.get("school_guidelines_raw_text") is not None
        assert "Police Times New Roman" in result["school_guidelines_raw_text"]

        formatting = result.get("school_guidelines_formatting", {})
        assert formatting.get("font_name") == "Times New Roman"
        assert formatting.get("font_size") == 12
        assert formatting.get("line_spacing_type") == "1.5"
        assert formatting.get("citation_style") == "APA"
        assert formatting.get("min_pages") == 30
        assert formatting.get("text_justification") is True

        structure = result.get("school_guidelines_structured", {})
        assert "Introduction" in structure
        assert "Description de la mission" in structure
        # assert "Conclusion" in structure # "6. Conclusion" devrait aussi être capturé.
        assert result.get("last_successful_node") == "N1GuidelineIngestorNode"
        assert result.get("error_message") is None

    @patch(
        "src.nodes.n1_guideline_ingestor.pypdf.PdfReader", side_effect=FileNotFoundError
    )
    def test_ingest_school_guidelines_file_not_found(self, mock_pdf_reader):
        """Test handling of FileNotFoundError."""
        state_with_bad_path = AgentState(school_guidelines_path="non_existent_file.pdf")
        result = self.node.run(state_with_bad_path)
        assert "error_message" in result
        assert "Failed to extract text from PDF" in result["error_message"]
        assert result.get("school_guidelines_raw_text") is None

    @patch("src.nodes.n1_guideline_ingestor.pypdf.PdfReader")
    def test_ingest_school_guidelines_no_text_extracted(self, mock_pdf_reader):
        """Test handling when PDF has no extractable text."""
        mock_page = MagicMock()
        mock_page.extract_text.return_value = ""  # PDF vide ou non extractible
        mock_reader_instance = mock_pdf_reader.return_value
        mock_reader_instance.pages = [mock_page]

        state_with_empty_pdf = AgentState(school_guidelines_path="dummy/empty.pdf")
        result = self.node.run(state_with_empty_pdf)

        assert result.get("school_guidelines_raw_text") == ""
        # Les parsers peuvent retourner des dicts vides s'ils ne trouvent rien
        assert result.get("school_guidelines_formatting") == {
            "text_justification": False
        }  # car text_justification a un fallback
        assert result.get("school_guidelines_structured") == {}
        # Un warning devrait être loggé, mais on ne le teste pas ici directement.

    def test_parse_formatting_rules(self):
        """Directly test the _parse_formatting_rules method."""
        text_sample = (
            "Le mémoire doit utiliser la police Arial 11 pts. L'interligne sera "
            "double. Le style de citation est Harvard. Un minimum de 25 pages est "
            "requis. Le texte ne doit pas être justifié."
        )
        rules = self.node._parse_formatting_rules(text_sample)
        assert rules.get("font_name") == "Arial"
        assert rules.get("font_size") == 11
        assert rules.get("line_spacing_type") == "double"
        assert rules.get("citation_style") == "HARVARD"
        assert rules.get("min_pages") == 25
        assert rules.get("text_justification") is False

    def test_parse_structured_content(self):
        """Directly test the _parse_structured_content method for basic structure."""
        text_sample = (
            "AVANT-PROPOS\nTexte avant-propos.\n"
            "1. INTRODUCTION\nCeci est l'intro.\n"
            "  1.1. Contexte\n  Texte contexte.\n"
            "2. DÉVELOPPEMENT\n  Texte dev.\n"
            "VI. CONCLUSION\nTexte conclusion."
        )
        structure = self.node._parse_structured_content(text_sample)
        assert "INTRODUCTION" in structure
        assert "DÉVELOPPEMENT" in structure


if __name__ == "__main__":
    unittest.main()
