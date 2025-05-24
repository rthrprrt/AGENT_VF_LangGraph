# tests/nodes/test_n2_rag_parts.py
import os
import shutil
from datetime import date
import pytest
from unittest.mock import patch, MagicMock

from langchain_core.documents import Document

from src.nodes.n2_journal_ingestor_anonymizer import (
    _load_raw_journal_entries,
    _chunk_entries_for_embedding,
    _save_or_update_faiss_store
)

class MockSettings:
    """Mock settings for tests."""
    embedding_model_name: str = "fastembed/BAAI/bge-small-en-v1.5"

mock_settings = MockSettings()

@pytest.fixture
def temp_journal_dir(tmp_path):
    """Crée un répertoire temporaire pour les fichiers journaux de test."""
    journal_dir = tmp_path / "journal_entries"
    journal_dir.mkdir()
    (journal_dir / "2023-10-25_entry1.txt").write_text(
        "Ceci est le contenu du journal 1.\nUne autre ligne."
    )
    (journal_dir / "2023-10-26_entry2.docx").write_text(
        "Contenu DOCX simulé pour entry2."
    )
    (journal_dir / "no_date_entry3.txt").write_text(
        "Entrée sans date dans le nom."
    )
    (journal_dir / "unsupported.pdf").write_text("Fichier PDF non supporté.")
    (journal_dir / "empty.txt").write_text("")
    return journal_dir

@pytest.fixture
def temp_faiss_dir(tmp_path):
    """Crée un répertoire temporaire pour le FAISS store de test."""
    faiss_dir = tmp_path / "faiss_store"
    faiss_dir.mkdir()
    return faiss_dir

@patch("src.nodes.n2_journal_ingestor_anonymizer.TextLoader")
@patch("src.nodes.n2_journal_ingestor_anonymizer.UnstructuredWordDocumentLoader")
def test_load_raw_journal_entries_success(
    mock_word_loader, mock_text_loader, temp_journal_dir
):
    """Tests successful loading of raw journal entries."""

    def text_loader_side_effect(file_path, encoding='utf-8'):
        mock_loader_instance = MagicMock()
        filename = os.path.basename(file_path)
        if filename == "2023-10-25_entry1.txt":
            mock_loader_instance.load.return_value = [
                Document(page_content="Ceci est le contenu du journal 1.\nUne autre ligne.")
            ]
        elif filename == "no_date_entry3.txt":
            mock_loader_instance.load.return_value = [
                Document(page_content="Entrée sans date dans le nom.")
            ]
        elif filename == "empty.txt":
            mock_loader_instance.load.return_value = [Document(page_content="")]
        else:
            mock_loader_instance.load.return_value = []
        return mock_loader_instance
    mock_text_loader.side_effect = text_loader_side_effect

    def word_loader_side_effect(file_path):
        mock_loader_instance = MagicMock()
        filename = os.path.basename(file_path)
        if filename == "2023-10-26_entry2.docx":
            mock_loader_instance.load.return_value = [
                Document(page_content="Contenu DOCX simulé pour entry2.")
            ]
        else:
            mock_loader_instance.load.return_value = []
        return mock_loader_instance
    mock_word_loader.side_effect = word_loader_side_effect

    entries = _load_raw_journal_entries(str(temp_journal_dir))

    assert len(entries) == 3

    entry1_found = False
    entry2_found = False
    entry3_found = False

    for entry in entries:
        if entry['source_file'] == "2023-10-25_entry1.txt":
            assert entry['raw_text'] == "Ceci est le contenu du journal 1.\nUne autre ligne."
            assert entry['entry_date'].isoformat() == "2023-10-25" # CORRIGÉ: 'date' -> 'entry_date'
            entry1_found = True
        elif entry['source_file'] == "2023-10-26_entry2.docx":
            assert entry['raw_text'] == "Contenu DOCX simulé pour entry2."
            assert entry['entry_date'].isoformat() == "2023-10-26" # CORRIGÉ: 'date' -> 'entry_date'
            entry2_found = True
        elif entry['source_file'] == "no_date_entry3.txt":
            assert entry['entry_date'] is None # CORRIGÉ: 'date' -> 'entry_date'
            assert entry['raw_text'] == "Entrée sans date dans le nom."
            entry3_found = True
    
    assert entry1_found and entry2_found and entry3_found
    assert mock_text_loader.call_count >= 3
    assert mock_word_loader.call_count >= 1

# ... (le reste du fichier test_n2_rag_parts.py comme dans la version précédente)
# ... (Je ne reproduis pas tout pour la concision)

def test_load_raw_journal_entries_non_existent_path():
    """Tests loading from a non-existent path."""
    entries = _load_raw_journal_entries("chemin/inexistant")
    assert len(entries) == 0

@patch('src.nodes.n2_journal_ingestor_anonymizer.TextLoader')
def test_load_raw_journal_entries_empty_file(
    mock_text_loader, temp_journal_dir
):
    """Tests that empty files are not added if they have no content."""
    empty_file_path = str(temp_journal_dir / "empty.txt")

    def specific_text_loader_side_effect(file_path, encoding='utf-8'):
        mock_loader_instance = MagicMock()
        if file_path == empty_file_path:
            mock_loader_instance.load.return_value = [Document(page_content="")]
        else:
            mock_loader_instance.load.return_value = [Document(page_content="autre contenu")]
        return mock_loader_instance
    mock_text_loader.side_effect = specific_text_loader_side_effect

    with patch(
        'src.nodes.n2_journal_ingestor_anonymizer.UnstructuredWordDocumentLoader'
    ) as mock_word_loader_local:
        mock_word_loader_local.return_value.load.return_value = []
        
        all_entries = _load_raw_journal_entries(str(temp_journal_dir))
        assert not any(e['source_file'] == "empty.txt" for e in all_entries)
        assert any(e['source_file'] == "2023-10-25_entry1.txt" for e in all_entries)


@patch('src.nodes.n2_journal_ingestor_anonymizer.RecursiveCharacterTextSplitter')
def test_chunk_entries_for_embedding_success(mock_splitter):
    """Tests successful chunking of processed entries."""
    mock_splitter_instance = mock_splitter.return_value
    mock_splitter_instance.split_text.side_effect = (
        lambda text: [text[:10], text[10:20]] if len(text) > 10 else [text]
    )
    processed_entries = [
        {
            'source_file': 'doc1.txt', 'entry_date': date(2023, 1, 1), # Utiliser date object
            'anonymized_text': 'Texte très long pour le document 1.'
        },
        {
            'source_file': 'doc2.txt', 'entry_date': date(2023, 1, 2), # Utiliser date object
            'anonymized_text': 'Court.'
        }
    ]
    documents = _chunk_entries_for_embedding(processed_entries)

    assert len(documents) == 3
    doc1_chunk1 = next(
        d for d in documents if d.metadata['source_document'] == 'doc1.txt' and
        d.metadata['chunk_index'] == 0
    )
    assert doc1_chunk1.page_content == 'Texte très'
    assert 'chunk_id' in doc1_chunk1.metadata
    assert doc1_chunk1.metadata['journal_date'] == '2023-01-01' # Doit être string ici

    doc2_chunk1 = next(d for d in documents if d.metadata['source_document'] == 'doc2.txt')
    assert doc2_chunk1.page_content == 'Court.'
    assert doc2_chunk1.metadata['chunk_index'] == 0

    mock_splitter.assert_called_once_with(
        chunk_size=1500,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    assert mock_splitter_instance.split_text.call_count == 2

def test_chunk_entries_for_embedding_no_anonymized_text():
    """Tests chunking when 'anonymized_text' is missing."""
    processed_entries = [
        {'source_file': 'doc1.txt', 'entry_date': date(2023,1,1),
         'raw_text': 'Texte original.'}
    ]
    documents = _chunk_entries_for_embedding(processed_entries)
    assert len(documents) == 0

def test_chunk_entries_for_embedding_empty_entries():
    """Tests chunking with an empty list of entries."""
    documents = _chunk_entries_for_embedding([])
    assert len(documents) == 0

@patch('src.nodes.n2_journal_ingestor_anonymizer.FAISS')
@patch('src.nodes.n2_journal_ingestor_anonymizer.FastEmbedEmbeddings')
@patch('src.nodes.n2_journal_ingestor_anonymizer.os.path.exists')
@patch('src.nodes.n2_journal_ingestor_anonymizer.os.makedirs')
@patch('src.nodes.n2_journal_ingestor_anonymizer.shutil.rmtree')
def test_save_or_update_faiss_store_recreate_new(
    mock_rmtree, mock_makedirs, mock_exists,
    mock_embeddings, mock_faiss, temp_faiss_dir
):
    """Tests creating a new FAISS store with recreate=True."""
    mock_exists.return_value = False
    mock_faiss_instance = mock_faiss.from_documents.return_value
    docs_for_faiss = [Document(page_content="chunk1", metadata={"id":1})]

    success = _save_or_update_faiss_store(
        str(temp_faiss_dir), docs_for_faiss,
        mock_settings.embedding_model_name, recreate_store=True
    )
    assert success
    mock_embeddings.assert_called_once_with(
        model_name=mock_settings.embedding_model_name
    )
    mock_faiss.from_documents.assert_called_once()
    mock_faiss_instance.save_local.assert_called_once_with(
        str(temp_faiss_dir)
    )
    mock_rmtree.assert_not_called()
    mock_makedirs.assert_called_with(str(temp_faiss_dir), exist_ok=True)

@patch('src.nodes.n2_journal_ingestor_anonymizer.FAISS')
@patch('src.nodes.n2_journal_ingestor_anonymizer.FastEmbedEmbeddings')
@patch('src.nodes.n2_journal_ingestor_anonymizer.os.path.exists')
@patch('src.nodes.n2_journal_ingestor_anonymizer.os.makedirs')
@patch('src.nodes.n2_journal_ingestor_anonymizer.shutil.rmtree')
def test_save_or_update_faiss_store_recreate_existing(
    mock_rmtree, mock_makedirs, mock_exists,
    mock_embeddings, mock_faiss, temp_faiss_dir
):
    """Tests recreating an existing FAISS store."""
    def exists_side_effect(path):
        if path == str(temp_faiss_dir): return True
        faiss_index_file = os.path.join(str(temp_faiss_dir), "index.faiss")
        if path == faiss_index_file:
            return False
        return False
    mock_exists.side_effect = exists_side_effect
    mock_faiss_instance = mock_faiss.from_documents.return_value
    docs_for_faiss = [Document(page_content="chunk1", metadata={"id":1})]

    success = _save_or_update_faiss_store(
        str(temp_faiss_dir), docs_for_faiss,
        mock_settings.embedding_model_name, recreate_store=True
    )
    assert success
    mock_rmtree.assert_called_once_with(str(temp_faiss_dir))
    mock_faiss.from_documents.assert_called_once()
    mock_faiss_instance.save_local.assert_called_once_with(str(temp_faiss_dir))
    mock_makedirs.assert_called_with(str(temp_faiss_dir), exist_ok=True)

@patch('src.nodes.n2_journal_ingestor_anonymizer.FAISS')
@patch('src.nodes.n2_journal_ingestor_anonymizer.FastEmbedEmbeddings')
@patch('src.nodes.n2_journal_ingestor_anonymizer.os.path.exists')
@patch('src.nodes.n2_journal_ingestor_anonymizer.shutil.rmtree')
def test_save_or_update_faiss_store_use_existing(
    mock_rmtree, mock_exists, mock_embeddings, mock_faiss, temp_faiss_dir
):
    """Tests using an existing FAISS store without recreation."""
    mock_exists.return_value = True
    docs_for_faiss = [Document(page_content="chunk1", metadata={"id":1})]
    success = _save_or_update_faiss_store(
        str(temp_faiss_dir), docs_for_faiss,
        mock_settings.embedding_model_name, recreate_store=False
    )
    assert success
    mock_embeddings.assert_called_once_with(
        model_name=mock_settings.embedding_model_name
    )
    mock_faiss.from_documents.assert_not_called()
    mock_rmtree.assert_not_called()

@patch('src.nodes.n2_journal_ingestor_anonymizer.FAISS')
@patch('src.nodes.n2_journal_ingestor_anonymizer.FastEmbedEmbeddings')
@patch('src.nodes.n2_journal_ingestor_anonymizer.os.path.exists')
def test_save_or_update_faiss_store_create_new_no_docs(
    mock_exists, mock_embeddings, mock_faiss, temp_faiss_dir
):
    """Tests creating a new store when no documents are provided."""
    mock_exists.return_value = False
    success = _save_or_update_faiss_store(
        str(temp_faiss_dir), [], mock_settings.embedding_model_name,
        recreate_store=False
    )
    assert not success
    mock_faiss.from_documents.assert_not_called()

@patch(
    'src.nodes.n2_journal_ingestor_anonymizer.FastEmbedEmbeddings',
    side_effect=Exception("Embedding init error")
)
def test_save_or_update_faiss_store_embedding_init_error(
    mock_embeddings, temp_faiss_dir
):
    """Tests error handling when embedding initialization fails."""
    success = _save_or_update_faiss_store(
        str(temp_faiss_dir), [Document(page_content="test")],
        "dummy_model", recreate_store=True
    )
    assert not success