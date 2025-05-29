# tests/tools/test_t1_journal_context_retriever.py
from unittest.mock import MagicMock, patch

import pytest
from langchain_community.vectorstores import FAISS  # Ajout de l'import FAISS
from langchain_core.documents import Document

from src.config import settings
from src.tools.t1_journal_context_retriever import (
    JournalContextRetrieverTool,
)

DEFAULT_EMBEDDING_MODEL_NAME_FOR_TEST = settings.embedding_model_name


@pytest.fixture()
def mock_faiss_index_for_tool():
    mock_index = MagicMock(spec=FAISS)
    mock_index.index = MagicMock()
    mock_index.index.ntotal = 2
    doc1 = Document(
        page_content="Contenu du chunk 1",
        metadata={"source": "doc1.txt", "chunk_id": "id1"},
    )
    doc2 = Document(
        page_content="Contenu du chunk 2",
        metadata={"source": "doc2.txt", "chunk_id": "id2"},
    )
    mock_index.similarity_search_with_score.return_value = [(doc1, 0.1), (doc2, 0.2)]
    return mock_index


@patch("src.tools.t1_journal_context_retriever.os.path.exists")
@patch("src.tools.t1_journal_context_retriever.FAISS.load_local")
@patch("src.tools.t1_journal_context_retriever.FastEmbedEmbeddings")
def test_journal_context_retriever_tool_success(
    mock_fastembed_embeddings,
    mock_faiss_load_local,
    mock_os_path_exists,
    mock_faiss_index_for_tool,
    tmp_path,
):
    mock_os_path_exists.return_value = True
    mock_embedding_instance = mock_fastembed_embeddings.return_value
    mock_faiss_load_local.return_value = mock_faiss_index_for_tool

    tool = JournalContextRetrieverTool(
        vector_store_path=str(tmp_path / "faiss_tool_test"),
        embedding_model_name=DEFAULT_EMBEDDING_MODEL_NAME_FOR_TEST,
    )
    results = tool._run(query_or_keywords="test query", k_retrieval_count=2)

    assert len(results) == 2
    assert results[0]["text"] == "Contenu du chunk 1"


@patch(
    "src.tools.t1_journal_context_retriever.FastEmbedEmbeddings",
    side_effect=Exception("Embedding init error"),
)
def test_journal_context_retriever_tool_embedding_error(
    mock_fastembed_embeddings, tmp_path
):
    tool = JournalContextRetrieverTool(
        vector_store_path=str(tmp_path / "faiss_emb_error"),
        embedding_model_name="error_model",
    )
    results = tool._run(query_or_keywords="test query")  # query_or_keywords est requis
    assert len(results) == 1
    assert "error" in results[0]
    # Le message d'erreur vient de _initialize_dependencies
    assert "Échec de l'initialisation des dépendances" in results[0]["error"]


@patch("src.tools.t1_journal_context_retriever.os.path.exists", return_value=False)
@patch("src.tools.t1_journal_context_retriever.FastEmbedEmbeddings")
def test_journal_context_retriever_tool_store_not_found(
    mock_fastembed_embeddings, mock_os_path_exists, tmp_path
):
    tool = JournalContextRetrieverTool(
        vector_store_path=str(tmp_path / "non_existent_faiss"),
        embedding_model_name=DEFAULT_EMBEDDING_MODEL_NAME_FOR_TEST,
    )
    results = tool._run(query_or_keywords="test query")
    assert len(results) == 1
    assert "error" in results[0]
    assert "Échec de l'initialisation des dépendances" in results[0]["error"]


@patch(
    "src.tools.t1_journal_context_retriever.FAISS.load_local",
    side_effect=Exception("FAISS load error"),
)
@patch("src.tools.t1_journal_context_retriever.FastEmbedEmbeddings")
@patch("src.tools.t1_journal_context_retriever.os.path.exists", return_value=True)
def test_journal_context_retriever_tool_faiss_load_error(
    mock_os_path_exists, mock_fastembed_embeddings, mock_faiss_load_local, tmp_path
):
    tool = JournalContextRetrieverTool(
        vector_store_path=str(tmp_path / "faiss_load_error"),
        embedding_model_name=DEFAULT_EMBEDDING_MODEL_NAME_FOR_TEST,
    )
    results = tool._run(query_or_keywords="test query")
    assert len(results) == 1
    assert "error" in results[0]
    assert "Échec de l'initialisation des dépendances" in results[0]["error"]


@patch("src.tools.t1_journal_context_retriever.FAISS.load_local")
@patch("src.tools.t1_journal_context_retriever.FastEmbedEmbeddings")
@patch("src.tools.t1_journal_context_retriever.os.path.exists")
def test_journal_context_retriever_tool_empty_index(  # Ce test passe car _perform_similarity_search retourne []
    mock_os_path_exists, mock_fastembed_embeddings, mock_faiss_load_local, tmp_path
):
    mock_os_path_exists.return_value = True
    empty_faiss_index = MagicMock(spec=FAISS)
    empty_faiss_index.index = MagicMock()
    empty_faiss_index.index.ntotal = 0
    mock_faiss_load_local.return_value = empty_faiss_index

    tool = JournalContextRetrieverTool(
        vector_store_path=str(tmp_path / "faiss_empty_tool"),
        embedding_model_name=DEFAULT_EMBEDDING_MODEL_NAME_FOR_TEST,
    )
    results = tool._run(query_or_keywords="test query")
    assert len(results) == 0  # Correct, car l'index est vide


# ... autres tests de T1 (no_results, args_schema, arun) ...
# Garder les autres tests de T1 tels qu'ils étaient dans la version "60 tests passés"
# que je vous ai fournie le 25 mai 2024 à 01:06 UTC
# (celle commençant par # tests/tools/test_t1_journal_context_retriever.py)
# Je ne les répète pas ici pour la concision mais ils sont importants.
# La version que vous avez utilisée pour les 60 tests passés est celle à conserver pour la fin du fichier.
# La principale modification ici était l'import de FAISS et les assertions sur les erreurs.


# Je vais juste inclure la fin du fichier `arun` pour être complet
@pytest.mark.asyncio()
@patch("src.tools.t1_journal_context_retriever.os.path.exists")
@patch("src.tools.t1_journal_context_retriever.FAISS.load_local")
@patch("src.tools.t1_journal_context_retriever.FastEmbedEmbeddings")
async def test_journal_context_retriever_tool_arun(
    mock_fastembed_embeddings,
    mock_faiss_load_local,
    mock_os_path_exists,
    mock_faiss_index_for_tool,
    tmp_path,
):
    mock_os_path_exists.return_value = True
    mock_fastembed_embeddings.return_value = MagicMock()
    mock_faiss_load_local.return_value = mock_faiss_index_for_tool

    tool = JournalContextRetrieverTool(
        vector_store_path=str(tmp_path / "faiss_async"),
        embedding_model_name=DEFAULT_EMBEDDING_MODEL_NAME_FOR_TEST,
    )

    k_test_value = 1
    doc1_async = Document(
        page_content="Async chunk 1", metadata={"source": "async1.txt"}
    )
    mock_faiss_index_for_tool.similarity_search_with_score.return_value = [
        (doc1_async, 0.3)
    ]

    results = await tool._arun(
        query_or_keywords="test query async", k_retrieval_count=k_test_value
    )

    mock_faiss_index_for_tool.similarity_search_with_score.assert_called_with(
        "test query async", k=k_test_value
    )

    assert len(results) == 1
    assert results[0]["text"] == "Async chunk 1"

    doc1 = Document(
        page_content="Contenu du chunk 1",
        metadata={"source": "doc1.txt", "chunk_id": "id1"},
    )
    doc2 = Document(
        page_content="Contenu du chunk 2",
        metadata={"source": "doc2.txt", "chunk_id": "id2"},
    )
    mock_faiss_index_for_tool.similarity_search_with_score.return_value = [
        (doc1, 0.1),
        (doc2, 0.2),
    ]
