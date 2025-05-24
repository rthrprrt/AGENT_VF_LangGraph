# tests/nodes/test_n2_rag_parts.py
import os
from datetime import date
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.nodes.n2_journal_ingestor_anonymizer import (
    _chunk_entries_for_embedding,
    _load_raw_journal_entries,
    _save_or_update_faiss_store,
)


class MockSettings:
    """Mock settings for tests."""

    embedding_model_name: str = "fastembed/BAAI/bge-small-en-v1.5"


mock_settings_instance = MockSettings()


@pytest.fixture()
def temp_journal_dir(tmp_path):
    """Creates a temporary directory for test journal files."""
    journal_dir = tmp_path / "journal_entries"
    journal_dir.mkdir()
    (journal_dir / "2023-10-25_entry1.txt").write_text(
        "Ceci est le contenu du journal 1.\nUne autre ligne."
    )
    (journal_dir / "2023-10-26_entry2.docx").write_text(
        "Contenu DOCX simulé pour entry2."
    )
    (journal_dir / "no_date_entry3.txt").write_text("Entrée sans date dans le nom.")
    (journal_dir / "unsupported.pdf").write_text("Fichier PDF non supporté.")
    (journal_dir / "empty.txt").write_text("")
    return str(journal_dir)


@pytest.fixture()
def temp_faiss_dir(tmp_path):
    """Creates a temporary directory for test FAISS store."""
    faiss_dir = tmp_path / "faiss_store"
    return str(faiss_dir)


@patch("src.nodes.n2_journal_ingestor_anonymizer.TextLoader")
@patch("src.nodes.n2_journal_ingestor_anonymizer.UnstructuredWordDocumentLoader")
def test_load_raw_journal_entries_success(
    mock_word_loader, mock_text_loader, temp_journal_dir
):
    """Tests successful loading of raw journal entries."""
    mock_txt_doc1 = Document(
        page_content="Ceci est le contenu du journal 1.\nUne autre ligne."
    )
    mock_txt_doc3 = Document(page_content="Entrée sans date dans le nom.")
    mock_docx_doc2 = Document(page_content="Contenu DOCX simulé pour entry2.")

    def text_loader_side_effect_precise(path, encoding):
        filename = os.path.basename(path)
        if filename == "2023-10-25_entry1.txt":
            return MagicMock(load=MagicMock(return_value=[mock_txt_doc1]))
        if filename == "no_date_entry3.txt":
            return MagicMock(load=MagicMock(return_value=[mock_txt_doc3]))
        if filename == "empty.txt":
            return MagicMock(load=MagicMock(return_value=[Document(page_content="")]))
        return MagicMock(load=MagicMock(return_value=[]))

    mock_text_loader.side_effect = text_loader_side_effect_precise
    mock_word_loader.return_value.load.return_value = [mock_docx_doc2]

    entries = _load_raw_journal_entries(temp_journal_dir)

    assert len(entries) == 3

    entry1 = next(
        (e for e in entries if e["source_file"] == "2023-10-25_entry1.txt"), None
    )
    assert entry1 is not None
    assert entry1["raw_text"] == "Ceci est le contenu du journal 1.\nUne autre ligne."
    assert entry1["entry_date"] == date(2023, 10, 25)

    entry2 = next(
        (e for e in entries if e["source_file"] == "2023-10-26_entry2.docx"), None
    )
    assert entry2 is not None
    assert entry2["raw_text"] == "Contenu DOCX simulé pour entry2."
    assert entry2["entry_date"] == date(2023, 10, 26)

    entry3 = next(
        (e for e in entries if e["source_file"] == "no_date_entry3.txt"), None
    )
    assert entry3 is not None
    assert entry3["entry_date"] is None
    assert entry3["raw_text"] == "Entrée sans date dans le nom."

    assert not any(e["source_file"] == "empty.txt" for e in entries)


def test_load_raw_journal_entries_non_existent_path():
    """Tests loading from a non-existent path."""
    entries = _load_raw_journal_entries("chemin/inexistant")
    assert len(entries) == 0


@patch("src.nodes.n2_journal_ingestor_anonymizer.TextLoader")
def test_load_raw_journal_entries_empty_file_behavior(
    mock_text_loader, temp_journal_dir
):
    """Tests that files loaded with no content are not included."""
    mock_text_loader_instance = mock_text_loader.return_value
    mock_text_loader_instance.load.return_value = [Document(page_content="")]

    with patch(
        "src.nodes.n2_journal_ingestor_anonymizer.UnstructuredWordDocumentLoader"
    ) as mock_word_loader_local:
        mock_word_loader_local.return_value.load.return_value = []
        empty_file_path = os.path.join(temp_journal_dir, "empty.txt")
        with open(empty_file_path, "w", encoding="utf-8") as f:
            f.write("")

        def specific_side_effect(path, encoding):
            if os.path.basename(path) == "empty.txt":
                return MagicMock(
                    load=MagicMock(return_value=[Document(page_content="")])
                )
            return MagicMock(load=MagicMock(return_value=[]))

        mock_text_loader.side_effect = specific_side_effect

        entries = _load_raw_journal_entries(str(temp_journal_dir))
        assert not any(e["source_file"] == "empty.txt" for e in entries)


def test_chunk_entries_for_embedding_success():
    """Tests successful chunking of processed entries."""
    processed_entries = [
        {
            "source_file": "doc1.txt",
            "entry_date": date(2023, 1, 1),
            "anonymized_text": (
                "Texte très long pour le document 1. Une phrase. " "Une autre phrase."
            ),
        },
        {
            "source_file": "doc2.txt",
            "entry_date": date(2023, 1, 2),
            "anonymized_text": "Court.",
        },
    ]
    documents = _chunk_entries_for_embedding(processed_entries)

    assert len(documents) == 2
    doc1_chunk1 = next(
        d for d in documents if d.metadata["source_document"] == "doc1.txt"
    )
    expected_content_doc1 = (
        "Texte très long pour le document 1. Une phrase. " "Une autre phrase."
    )
    assert doc1_chunk1.page_content == expected_content_doc1
    assert "chunk_id" in doc1_chunk1.metadata
    assert doc1_chunk1.metadata["journal_date"] == "2023-01-01"

    doc2_chunk1 = next(
        d for d in documents if d.metadata["source_document"] == "doc2.txt"
    )
    assert doc2_chunk1.page_content == "Court."


def test_chunk_entries_for_embedding_no_anonymized_text():
    """Tests chunking when 'anonymized_text' is missing."""
    processed_entries = [
        {
            "source_file": "doc1.txt",
            "entry_date": date(2023, 1, 1),
            "raw_text": "Texte original non anonymisé.",
        }
    ]
    documents = _chunk_entries_for_embedding(processed_entries)
    assert len(documents) == 0


def test_chunk_entries_for_embedding_empty_entries():
    """Tests chunking with an empty list of entries."""
    documents = _chunk_entries_for_embedding([])
    assert len(documents) == 0


@patch("src.nodes.n2_journal_ingestor_anonymizer.FAISS")
@patch("src.nodes.n2_journal_ingestor_anonymizer.FastEmbedEmbeddings")
@patch("src.nodes.n2_journal_ingestor_anonymizer.os.path.exists")
@patch("src.nodes.n2_journal_ingestor_anonymizer.os.makedirs")
@patch("src.nodes.n2_journal_ingestor_anonymizer.shutil.rmtree")
def test_save_or_update_faiss_store_recreate_new(
    mock_rmtree,
    mock_makedirs,
    mock_os_path_exists,
    mock_embeddings,
    mock_faiss,
    temp_faiss_dir,
):
    """Tests creating a new FAISS store with recreate=True."""
    mock_os_path_exists.return_value = False
    mock_faiss_instance = mock_faiss.from_documents.return_value
    docs_for_faiss = [Document(page_content="chunk1", metadata={"id": 1})]

    success = _save_or_update_faiss_store(
        temp_faiss_dir,
        docs_for_faiss,
        mock_settings_instance.embedding_model_name,
        recreate_store=True,
    )
    assert success
    mock_embeddings.assert_called_once_with(
        model_name=mock_settings_instance.embedding_model_name
    )
    mock_faiss.from_documents.assert_called_once_with(
        docs_for_faiss, mock_embeddings.return_value
    )
    mock_faiss_instance.save_local.assert_called_once_with(temp_faiss_dir)
    mock_rmtree.assert_not_called()
    mock_makedirs.assert_called_with(temp_faiss_dir, exist_ok=True)


@patch("src.nodes.n2_journal_ingestor_anonymizer.FAISS")
@patch("src.nodes.n2_journal_ingestor_anonymizer.FastEmbedEmbeddings")
@patch("src.nodes.n2_journal_ingestor_anonymizer.os.path.exists")
@patch("src.nodes.n2_journal_ingestor_anonymizer.os.makedirs")
@patch("src.nodes.n2_journal_ingestor_anonymizer.shutil.rmtree")
def test_save_or_update_faiss_store_recreate_existing(
    mock_rmtree,
    mock_makedirs,
    mock_os_path_exists,
    mock_embeddings,
    mock_faiss,
    temp_faiss_dir,
):
    """Tests recreating an existing FAISS store."""

    def exists_side_effect(path):
        if path == temp_faiss_dir:
            return True
        faiss_index_file = os.path.join(temp_faiss_dir, "index.faiss")
        if path == faiss_index_file:
            return False
        return False

    mock_os_path_exists.side_effect = exists_side_effect

    mock_faiss_instance = mock_faiss.from_documents.return_value
    docs_for_faiss = [Document(page_content="chunk1", metadata={"id": 1})]

    success = _save_or_update_faiss_store(
        temp_faiss_dir,
        docs_for_faiss,
        mock_settings_instance.embedding_model_name,
        recreate_store=True,
    )
    assert success
    mock_rmtree.assert_called_once_with(temp_faiss_dir)
    mock_faiss.from_documents.assert_called_once()
    mock_faiss_instance.save_local.assert_called_once_with(temp_faiss_dir)
    mock_makedirs.assert_called_with(temp_faiss_dir, exist_ok=True)


@patch("src.nodes.n2_journal_ingestor_anonymizer.FAISS")
@patch("src.nodes.n2_journal_ingestor_anonymizer.FastEmbedEmbeddings")
@patch("src.nodes.n2_journal_ingestor_anonymizer.os.path.exists")
@patch("src.nodes.n2_journal_ingestor_anonymizer.shutil.rmtree")
def test_save_or_update_faiss_store_use_existing(
    mock_rmtree, mock_os_path_exists, mock_embeddings, mock_faiss, temp_faiss_dir
):
    """Tests using an existing FAISS store without recreation."""
    mock_os_path_exists.return_value = True  # Store and index.faiss exist
    docs_for_faiss = [Document(page_content="chunk1", metadata={"id": 1})]
    success = _save_or_update_faiss_store(
        temp_faiss_dir,
        docs_for_faiss,
        mock_settings_instance.embedding_model_name,
        recreate_store=False,
    )
    assert success
    mock_embeddings.assert_called_once_with(
        model_name=mock_settings_instance.embedding_model_name
    )
    mock_faiss.from_documents.assert_not_called()
    mock_rmtree.assert_not_called()


@patch("src.nodes.n2_journal_ingestor_anonymizer.FAISS")
@patch("src.nodes.n2_journal_ingestor_anonymizer.FastEmbedEmbeddings")
@patch("src.nodes.n2_journal_ingestor_anonymizer.os.path.exists")
def test_save_or_update_faiss_store_create_new_no_docs(
    mock_os_path_exists, mock_embeddings, mock_faiss, temp_faiss_dir
):
    """Tests creating a new store when no documents are provided."""
    mock_os_path_exists.return_value = False
    success = _save_or_update_faiss_store(
        temp_faiss_dir,
        [],
        mock_settings_instance.embedding_model_name,
        recreate_store=False,  # Ligne 304 (E501) - Ligne OK
    )
    assert (
        not success
    )  # La fonction retourne False si pas de docs et store n'existait pas
    mock_embeddings.assert_called_once()
    mock_faiss.from_documents.assert_not_called()


@patch(
    "src.nodes.n2_journal_ingestor_anonymizer.FastEmbedEmbeddings",
    side_effect=Exception("Embedding init error"),  # Ligne 312 (E501) - OK
)
def test_save_or_update_faiss_store_embedding_init_error(  # Ligne 314 (E501) - OK
    mock_embeddings,
    temp_faiss_dir,  # Ligne 315 (E501) - OK
):
    """Tests error handling when embedding initialization fails."""
    docs = [Document(page_content="test")]
    success = _save_or_update_faiss_store(
        temp_faiss_dir, docs, "dummy_model", recreate_store=True
    )
    assert not success
