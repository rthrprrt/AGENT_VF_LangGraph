# tests/nodes/test_n2_rag_parts.py
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.config import settings

# Importer la classe du nœud et les fonctions utilitaires si elles sont DANS le module du nœud
from src.nodes.n2_journal_ingestor_anonymizer import (
    N2JournalIngestorAnonymizerNode,
    _load_raw_journal_entries_from_files,  # La fonction principale de chargement
)

DEFAULT_EMBEDDING_MODEL_FOR_N2_TEST = settings.embedding_model_name


@pytest.fixture()
def n2_node_instance():
    return N2JournalIngestorAnonymizerNode(
        chunk_size=100, chunk_overlap=10
    )  # Pour tester le chunking


@pytest.fixture()
def temp_journal_dir(tmp_path: Path) -> Path:
    journal_dir = tmp_path / "journal_entries"
    journal_dir.mkdir()
    return journal_dir


@pytest.fixture()
def temp_vector_store_dir(tmp_path: Path) -> Path:
    vs_dir = tmp_path / "faiss_store_n2"
    return vs_dir


def create_dummy_file(path: Path, content: str):
    path.write_text(content, encoding="utf-8")


# --- Tests pour le chargement des journaux ---
# Utiliser les fonctions importées depuis le module n2 si elles sont globales,
# ou les méthodes de l'instance n2_node_instance si elles sont des méthodes.
# Ici, on va tester _load_raw_journal_entries_from_files qui est la fonction utilisée par N2.run()


@patch("src.nodes.n2_journal_ingestor_anonymizer._load_single_journal_file")
def test_load_raw_journal_entries_success(mock_load_single, temp_journal_dir: Path):
    # _load_single_journal_file retourne une liste de Document Langchain
    doc1_content = "Contenu du journal 1.\nUne autre ligne."
    doc2_content = "Contenu DOCX simulé pour entry2."
    doc3_content = "Entrée sans date dans le nom."

    def load_single_side_effect(file_path: Path):
        if file_path.name == "2023-10-25_entry1.txt":
            return [
                Document(page_content=doc1_content, metadata={"source": str(file_path)})
            ]
        if file_path.name == "2023-10-26_entry2.docx":
            return [
                Document(page_content=doc2_content, metadata={"source": str(file_path)})
            ]
        if file_path.name == "no_date_entry3.txt":
            return [
                Document(page_content=doc3_content, metadata={"source": str(file_path)})
            ]
        if file_path.name == "empty.txt":
            return [
                Document(page_content="")
            ]  # Simule un fichier vide qui donne un Document vide
        return []  # Pour les fichiers non supportés ou erreurs

    mock_load_single.side_effect = load_single_side_effect

    create_dummy_file(temp_journal_dir / "2023-10-25_entry1.txt", "dummy")
    create_dummy_file(temp_journal_dir / "2023-10-26_entry2.docx", "dummy")
    create_dummy_file(temp_journal_dir / "no_date_entry3.txt", "dummy")
    create_dummy_file(temp_journal_dir / "empty.txt", "")
    create_dummy_file(temp_journal_dir / "unsupported.pdf", "dummy")

    entries = _load_raw_journal_entries_from_files(str(temp_journal_dir))

    assert len(entries) == 3
    assert entries[0]["source_file"] == "2023-10-25_entry1.txt"
    assert entries[0]["raw_text"] == doc1_content
    assert entries[0]["date_str"] == "2023-10-25"
    assert entries[1]["source_file"] == "2023-10-26_entry2.docx"
    assert entries[1]["raw_text"] == doc2_content
    assert entries[1]["date_str"] == "2023-10-26"
    assert entries[2]["source_file"] == "no_date_entry3.txt"
    assert entries[2]["date_str"] == "Date inconnue"


def test_load_raw_journal_entries_non_existent_path():
    entries = _load_raw_journal_entries_from_files("chemin/inexistant")
    assert len(entries) == 0


@patch("src.nodes.n2_journal_ingestor_anonymizer._load_single_journal_file")
def test_load_raw_journal_entries_loader_exception(
    mock_load_single, temp_journal_dir: Path, caplog
):
    mock_load_single.side_effect = Exception("Erreur de chargement simulée")
    create_dummy_file(temp_journal_dir / "entry1.txt", "contenu")

    entries = _load_raw_journal_entries_from_files(str(temp_journal_dir))
    assert len(entries) == 0
    assert (
        "Échec du chargement de _load_single_journal_file pour entry1.txt"
        in caplog.text
    )
    assert "Erreur de chargement simulée" in caplog.text


# --- Tests pour _chunk_entries_for_embedding ---
def test_chunk_entries_for_embedding_success(
    n2_node_instance: N2JournalIngestorAnonymizerNode,
):
    processed_entries = [
        {
            "anonymized_text": "Texte très long pour le document 1. Une phrase. Une autre phrase.",
            "source_file": "doc1.txt",
            "date_str": "2023-01-01",
        },
        {
            "anonymized_text": "Court.",
            "source_file": "doc2.txt",
            "date_str": "2023-01-02",
        },
    ]
    n2_node_instance.chunk_size = 50
    docs = n2_node_instance._chunk_entries_for_embedding(processed_entries)

    assert len(docs) > 1
    assert docs[0].metadata["source_document"] == "doc1.txt"
    assert docs[0].metadata["journal_date"] == "2023-01-01"
    assert docs[-1].metadata["source_document"] == "doc2.txt"
    assert docs[-1].metadata["journal_date"] == "2023-01-02"


def test_chunk_entries_for_embedding_no_anonymized_text(
    n2_node_instance: N2JournalIngestorAnonymizerNode, caplog
):
    processed_entries = [
        {"anonymized_text": None, "source_file": "doc1.txt"},
        {"anonymized_text": "  ", "source_file": "doc2.txt"},
    ]
    docs = n2_node_instance._chunk_entries_for_embedding(processed_entries)
    assert len(docs) == 0
    assert "Aucun 'anonymized_text' pour l'entrée source: doc1.txt" in caplog.text
    assert "Aucun 'anonymized_text' pour l'entrée source: doc2.txt" in caplog.text


@patch("src.nodes.n2_journal_ingestor_anonymizer.FAISS")
@patch("src.nodes.n2_journal_ingestor_anonymizer.FastEmbedEmbeddings")
def test_save_or_update_faiss_store_recreate_new(
    mock_fastembed: MagicMock,
    mock_faiss: MagicMock,
    n2_node_instance: N2JournalIngestorAnonymizerNode,
    temp_vector_store_dir: Path,
):
    mock_embeddings_instance = mock_fastembed.return_value
    mock_faiss_db_instance = mock_faiss.from_documents.return_value

    docs = [Document(page_content="test")]
    success = n2_node_instance._save_or_update_faiss_store(
        docs, str(temp_vector_store_dir), DEFAULT_EMBEDDING_MODEL_FOR_N2_TEST, True
    )
    assert success is True
    mock_faiss.from_documents.assert_called_once_with(docs, mock_embeddings_instance)
    mock_faiss_db_instance.save_local.assert_called_once_with(
        folder_path=str(temp_vector_store_dir)
    )


@patch("src.nodes.n2_journal_ingestor_anonymizer.FAISS")
@patch("src.nodes.n2_journal_ingestor_anonymizer.FastEmbedEmbeddings")
@patch("src.nodes.n2_journal_ingestor_anonymizer.Path.exists")
@patch("src.nodes.n2_journal_ingestor_anonymizer.Path.unlink")
@patch("src.nodes.n2_journal_ingestor_anonymizer.Path.rmdir")
def test_save_or_update_faiss_store_recreate_existing(
    mock_rmdir: MagicMock,
    mock_unlink: MagicMock,
    mock_path_exists: MagicMock,
    mock_fastembed: MagicMock,
    mock_faiss: MagicMock,
    n2_node_instance: N2JournalIngestorAnonymizerNode,
    temp_vector_store_dir: Path,
):
    mock_path_exists.return_value = True

    with patch.object(Path, "iterdir", return_value=iter([])):
        with patch.object(Path, "is_file", return_value=True):
            mock_embeddings_instance = mock_fastembed.return_value
            mock_faiss_db_instance = mock_faiss.from_documents.return_value

            docs = [Document(page_content="test")]
            success = n2_node_instance._save_or_update_faiss_store(
                docs,
                str(temp_vector_store_dir),
                DEFAULT_EMBEDDING_MODEL_FOR_N2_TEST,
                True,
            )
            assert success is True
            mock_faiss.from_documents.assert_called_once_with(
                docs, mock_embeddings_instance
            )
            mock_faiss_db_instance.save_local.assert_called_once_with(
                folder_path=str(temp_vector_store_dir)
            )
            # mock_unlink devrait avoir été appelé pour index.faiss et index.pkl
            # si la logique interne de _save_or_update_faiss_store
            # appelle (vector_store_path / "index.faiss").exists() et que cela retourne True.
            # Le mock_path_exists actuel est global.
            # Pour l'instant, nous allons supposer que si mock_path_exists (pour le dossier) est True,
            # les fichiers internes existent aussi pour la logique de suppression.
            # Ce test est plus axé sur la recréation que sur la finesse de la suppression.


@patch("src.nodes.n2_journal_ingestor_anonymizer.FAISS")
@patch("src.nodes.n2_journal_ingestor_anonymizer.FastEmbedEmbeddings")
@patch("src.nodes.n2_journal_ingestor_anonymizer.Path.exists")
def test_save_or_update_faiss_store_use_existing(
    mock_path_exists: MagicMock,
    mock_fastembed: MagicMock,
    mock_faiss: MagicMock,
    n2_node_instance: N2JournalIngestorAnonymizerNode,
    temp_vector_store_dir: Path,
):
    # Correction du side_effect: il doit juste retourner True pour ce test spécifique
    # où l'on suppose que les fichiers et le répertoire existent.
    # La complexité de la signature n'est pas le problème ici, mais ce que le mock doit retourner.
    # Pour que la condition `index_file.exists() and pkl_file.exists() and index_file.stat().st_size > 0`
    # soit vraie, `mock_path_exists` doit retourner True quand on l'appelle sur index_file et pkl_file.

    # Mock pour Path(...).exists()
    # Ce mock sera appelé pour vector_store_path.exists(), index_file.exists(), pkl_file.exists()
    mock_path_exists.return_value = True

    mock_index_file_stat = MagicMock()
    mock_index_file_stat.st_size = 100

    with patch.object(Path, "stat", return_value=mock_index_file_stat):
        mock_embeddings_instance = mock_fastembed.return_value
        mock_faiss_db_instance = mock_faiss.load_local.return_value

        docs = [Document(page_content="new doc")]
        success = n2_node_instance._save_or_update_faiss_store(
            docs, str(temp_vector_store_dir), DEFAULT_EMBEDDING_MODEL_FOR_N2_TEST, False
        )
        assert (
            success is True
        )  # L'erreur initiale provenait du TypeError qui faisait retourner False
        mock_faiss.load_local.assert_called_once_with(
            folder_path=str(temp_vector_store_dir),
            embeddings=mock_embeddings_instance,
            allow_dangerous_deserialization=True,
        )
        mock_faiss_db_instance.add_documents.assert_called_once_with(docs)
        mock_faiss_db_instance.save_local.assert_called_once_with(
            folder_path=str(temp_vector_store_dir)
        )


@patch("src.nodes.n2_journal_ingestor_anonymizer.FAISS")
@patch("src.nodes.n2_journal_ingestor_anonymizer.FastEmbedEmbeddings")
def test_save_or_update_faiss_store_create_new_no_docs_recreate_false(
    mock_fastembed: MagicMock,
    mock_faiss: MagicMock,
    n2_node_instance: N2JournalIngestorAnonymizerNode,
    temp_vector_store_dir: Path,
    caplog,
):
    mock_embeddings_instance = mock_fastembed.return_value
    success = n2_node_instance._save_or_update_faiss_store(
        [],
        str(temp_vector_store_dir),
        DEFAULT_EMBEDDING_MODEL_FOR_N2_TEST,
        False,
    )
    assert success is True
    assert "Aucun document à indexer." in caplog.text
    mock_faiss.from_documents.assert_not_called()
    mock_faiss.load_local.assert_not_called()


@patch(
    "src.nodes.n2_journal_ingestor_anonymizer.FastEmbedEmbeddings",
    side_effect=Exception("Embedding init error"),
)
def test_save_or_update_faiss_store_embedding_init_error(
    mock_fastembed: MagicMock,
    n2_node_instance: N2JournalIngestorAnonymizerNode,
    temp_vector_store_dir: Path,
):
    success = n2_node_instance._save_or_update_faiss_store(
        [Document(page_content="test")], str(temp_vector_store_dir), "bad_model", True
    )
    assert success is False
