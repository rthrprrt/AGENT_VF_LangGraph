# tests/tools/test_t1_journal_context_retriever.py
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.tools.t1_journal_context_retriever import (
    JournalContextRetrieverArgs,
    JournalContextRetrieverTool,
)


class MockToolSettings:
    """Mock settings for tool tests."""

    embedding_model_name: str = "fastembed/BAAI/bge-small-en-v1.5"


mock_tool_settings_instance = MockToolSettings()


@pytest.fixture()
def mock_faiss_index_for_tool():
    """Creates a mocked FAISS index instance."""
    mock_index = MagicMock()
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


@patch("src.tools.t1_journal_context_retriever.FAISS.load_local")
@patch("src.tools.t1_journal_context_retriever.FastEmbedEmbeddings")
@patch("src.tools.t1_journal_context_retriever.os.path.exists")
def test_journal_context_retriever_tool_success(
    mock_os_path_exists,
    mock_fastembed_embeddings,
    mock_faiss_load_local,
    mock_faiss_index_for_tool,
    tmp_path,
):
    """Tests successful run of the retriever tool."""
    mock_os_path_exists.return_value = True
    mock_embedding_instance = mock_fastembed_embeddings.return_value
    mock_faiss_load_local.return_value = mock_faiss_index_for_tool

    tool = JournalContextRetrieverTool(
        vector_store_path=str(tmp_path / "faiss_tool_test"),
        embedding_model_name=mock_tool_settings_instance.embedding_model_name,
    )
    results = tool._run(query_or_keywords="test query", k_retrieval_count=2)

    assert len(results) == 2
    assert results[0]["text"] == "Contenu du chunk 1"
    assert results[0]["metadata"]["source"] == "doc1.txt"
    assert results[0]["score"] == 0.1

    mock_fastembed_embeddings.assert_called_once_with(
        model_name=mock_tool_settings_instance.embedding_model_name
    )
    # Ligne 80 (E501) - Coupée
    mock_faiss_load_local.assert_called_once_with(
        str(tmp_path / "faiss_tool_test"),
        mock_embedding_instance,
        allow_dangerous_deserialization=True,
    )
    mock_faiss_index_for_tool.similarity_search_with_score.assert_called_once_with(
        "test query", k=2
    )


# ... (le reste du fichier test_t1_journal_context_retriever.py comme avant)
@patch(
    "src.tools.t1_journal_context_retriever.FastEmbedEmbeddings",
    side_effect=Exception("Embedding init error"),
)
def test_journal_context_retriever_tool_embedding_error(
    mock_fastembed_embeddings, tmp_path
):
    """Tests tool behavior when embedding initialization fails."""
    tool = JournalContextRetrieverTool(
        vector_store_path=str(tmp_path / "faiss_tool_test"),
        embedding_model_name="error_model",
    )
    results = tool._run(query_or_keywords="test query")
    assert len(results) == 1
    assert "error" in results[0]
    assert "Échec init embedding model" in results[0]["error"]


@patch("src.tools.t1_journal_context_retriever.os.path.exists", return_value=False)
@patch("src.tools.t1_journal_context_retriever.FastEmbedEmbeddings")
def test_journal_context_retriever_tool_store_not_found(
    mock_fastembed_embeddings, mock_os_path_exists, tmp_path
):
    """Tests tool behavior when the vector store is not found."""
    tool = JournalContextRetrieverTool(
        vector_store_path=str(tmp_path / "non_existent_faiss"),
        embedding_model_name=mock_tool_settings_instance.embedding_model_name,
    )
    results = tool._run(query_or_keywords="test query")
    assert len(results) == 1
    assert "error" in results[0]
    expected_error_part = "Vector store non trouvé"
    assert expected_error_part in results[0]["error"]
    mock_fastembed_embeddings.assert_called_once()


@patch(
    "src.tools.t1_journal_context_retriever.FAISS.load_local",
    side_effect=Exception("FAISS load error"),
)
@patch("src.tools.t1_journal_context_retriever.FastEmbedEmbeddings")
@patch("src.tools.t1_journal_context_retriever.os.path.exists", return_value=True)
def test_journal_context_retriever_tool_faiss_load_error(
    mock_os_path_exists, mock_fastembed_embeddings, mock_faiss_load_local, tmp_path
):
    """Tests tool behavior when FAISS index loading fails."""
    tool = JournalContextRetrieverTool(
        vector_store_path=str(tmp_path / "faiss_error"),
        embedding_model_name=mock_tool_settings_instance.embedding_model_name,
    )
    results = tool._run(query_or_keywords="test query")
    assert len(results) == 1
    assert "error" in results[0]
    assert "Échec chargement index FAISS" in results[0]["error"]


@patch("src.tools.t1_journal_context_retriever.FAISS.load_local")
@patch("src.tools.t1_journal_context_retriever.FastEmbedEmbeddings")
@patch("src.tools.t1_journal_context_retriever.os.path.exists")
def test_journal_context_retriever_tool_empty_index(
    mock_os_path_exists, mock_fastembed_embeddings, mock_faiss_load_local, tmp_path
):
    """Tests tool behavior with an empty FAISS index."""
    mock_os_path_exists.return_value = True
    empty_faiss_index = MagicMock()
    empty_faiss_index.index = MagicMock()
    empty_faiss_index.index.ntotal = 0
    mock_faiss_load_local.return_value = empty_faiss_index

    tool = JournalContextRetrieverTool(
        vector_store_path=str(tmp_path / "faiss_empty_tool"),
        embedding_model_name=mock_tool_settings_instance.embedding_model_name,
    )
    results = tool._run(query_or_keywords="test query")
    assert len(results) == 0


@patch("src.tools.t1_journal_context_retriever.FAISS.load_local")
@patch("src.tools.t1_journal_context_retriever.FastEmbedEmbeddings")
@patch("src.tools.t1_journal_context_retriever.os.path.exists")
def test_journal_context_retriever_tool_no_results(
    mock_os_path_exists,
    mock_fastembed_embeddings,
    mock_faiss_load_local,
    mock_faiss_index_for_tool,
    tmp_path,
):
    """Tests tool behavior when no relevant documents are found."""
    mock_os_path_exists.return_value = True
    mock_faiss_load_local.return_value = mock_faiss_index_for_tool
    mock_faiss_index_for_tool.similarity_search_with_score.return_value = []

    tool = JournalContextRetrieverTool(
        vector_store_path=str(tmp_path / "faiss_no_results"),
        embedding_model_name=mock_tool_settings_instance.embedding_model_name,
    )
    results = tool._run(query_or_keywords="requête sans résultats")
    assert len(results) == 0


def test_journal_context_retriever_tool_args_schema():
    """Tests the arguments schema of the retriever tool."""
    tool = JournalContextRetrieverTool(
        vector_store_path="dummy", embedding_model_name="dummy"
    )
    assert tool.args_schema == JournalContextRetrieverArgs
    schema_props = JournalContextRetrieverArgs.schema()["properties"]
    assert "query_or_keywords" in schema_props
    desc_query = "La requête ou les mots-clés à rechercher dans le journal."
    assert schema_props["query_or_keywords"]["description"] == desc_query
    assert "k_retrieval_count" in schema_props
    assert schema_props["k_retrieval_count"]["default"] == 3


@patch("src.tools.t1_journal_context_retriever.FAISS.load_local")
@patch("src.tools.t1_journal_context_retriever.FastEmbedEmbeddings")
@patch("src.tools.t1_journal_context_retriever.os.path.exists")
async def test_journal_context_retriever_tool_arun(
    mock_os_path_exists,
    mock_fastembed_embeddings,
    mock_faiss_load_local,
    mock_faiss_index_for_tool,
    tmp_path,
):
    """Tests the asynchronous run method of the retriever tool."""
    mock_os_path_exists.return_value = True
    mock_faiss_load_local.return_value = mock_faiss_index_for_tool

    tool = JournalContextRetrieverTool(
        vector_store_path=str(tmp_path / "faiss_async"),
        embedding_model_name=mock_tool_settings_instance.embedding_model_name,
    )
    results = await tool._arun(
        query_or_keywords="test query async", k_retrieval_count=1
    )
    assert len(results) == 2
    mock_faiss_index_for_tool.similarity_search_with_score.assert_called_with(
        "test query async", k=1
    )
