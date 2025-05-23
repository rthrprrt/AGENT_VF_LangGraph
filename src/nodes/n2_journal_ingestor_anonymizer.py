# src/nodes/n2_journal_ingestor_anonymizer.py
import logging
import os
from typing import Any

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Utiliser les vrais imports quand intégrés
from src.state import AgentState

logger = logging.getLogger(__name__)


def load_raw_journal_entries(journal_path: str) -> list[dict[str, Any]]:
    """Charge le contenu brut des fichiers .txt et .docx du journal.

    Args:
        journal_path: Chemin vers le dossier contenant les fichiers journal

    Returns:
        Liste des entrées de journal avec leur contenu et métadonnées
    """
    entries: list[dict[str, Any]] = []
    if not os.path.isdir(journal_path):
        logger.error(f"Journal path {journal_path} not found or not a directory.")
        return entries

    for filename in os.listdir(journal_path):
        file_path = os.path.join(journal_path, filename)
        entry_date: str | None = None
        # entry_date = filename.split('.')[0] # Exemple pour regex \d{4}-\d{2}-\d{2}

        raw_text_content = ""
        try:
            if filename.endswith(".txt"):
                loader = TextLoader(file_path, encoding="utf-8")
                docs = loader.load()
                if docs:
                    raw_text_content = docs[0].page_content
            elif filename.endswith(".docx"):
                loader = UnstructuredWordDocumentLoader(file_path)
                docs = loader.load()
                if docs:
                    raw_text_content = docs[0].page_content
            else:
                continue

            entries.append(
                {
                    "source_document": filename,
                    "date": entry_date,
                    "text": raw_text_content,
                }
            )
        # Correction BLE001: Utilisation de logger.error avec exc_info
        except FileNotFoundError:
            logger.error(f"File not found: {filename}")
        except Exception:
            logger.error(f"Error loading file {filename}", exc_info=True)

    return entries


def chunk_text(text_entries: list[dict[str, Any]]) -> list[Document]:
    """Applique le chunking avancé et prépare les documents pour l'embedding.

    Args:
        text_entries: Liste des entrées de texte à chunker

    Returns:
        Liste des documents chunkés prêts pour l'embedding
    """
    chunk_size = 1500
    chunk_overlap = 200
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

    all_docs_for_faiss: list[Document] = []
    for entry_index, entry in enumerate(text_entries):
        text_to_chunk = entry.get("anonymized_text", entry.get("text", ""))
        if not text_to_chunk:
            continue

        chunks = text_splitter.split_text(text_to_chunk)

        for i, chunk_text in enumerate(chunks):
            # Correction E501: f-string coupée
            doc_id_part = entry.get("source_document", f"doc_{entry_index}")
            chunk_id = f"{doc_id_part}_chunk_{i}"
            metadata = {
                "source_document": entry.get(
                    "source_document", f"unknown_doc_{entry_index}"
                ),
                "journal_date": entry.get("date", None),
                "chunk_index": i,
                "chunk_id": chunk_id,
            }
            all_docs_for_faiss.append(
                Document(page_content=chunk_text, metadata=metadata)
            )

    return all_docs_for_faiss


def journal_ingestor_anonymizer_node(state: AgentState) -> AgentState:
    """
    Ingère, anonymise, chunke, embed et gère FAISS.

    Ce nœud orchestre le chargement du journal, son anonymisation
    (par une logique appelée séparément mais conceptuellement ici),
    puis le traitement RAG (chunking, embedding, stockage vectoriel).
    """
    logger.info("N2: Journal Ingestion, Anonymization, and Vectorization...")

    journal_path = state.journal_path
    vector_store_path = state.vector_store_path

    if not journal_path:
        state.error_message = "Journal path not provided."
        logger.error(state.error_message)
        return state

    try:
        # Étape 1: Charger les entrées brutes du journal
        logger.info(f"Loading journal entries from: {journal_path}")
        raw_entries = load_raw_journal_entries(journal_path)

        if not raw_entries:
            state.error_message = "No journal entries found or loaded."
            logger.error(state.error_message)
            return state

        logger.info(f"Loaded {len(raw_entries)} journal entries.")

        # Étape 2: Anonymisation (placeholder - à implémenter)
        # Pour l'instant, on copie simplement le texte original
        anonymized_entries = []
        for entry in raw_entries:
            anonymized_entry = entry.copy()
            # Placeholder pour l'anonymisation
            anonymized_entry["anonymized_text"] = entry["text"]
            anonymized_entries.append(anonymized_entry)

        logger.info("Anonymization completed (placeholder).")

        # Étape 3: Chunking
        logger.info("Starting text chunking...")
        chunked_docs = chunk_text(anonymized_entries)

        if not chunked_docs:
            state.error_message = "No chunks created from journal entries."
            logger.error(state.error_message)
            return state

        logger.info(f"Created {len(chunked_docs)} text chunks.")

        # Étape 4: Embedding et stockage vectoriel
        logger.info("Creating embeddings and vector store...")

        embeddings = FastEmbedEmbeddings()
        vector_store = FAISS.from_documents(chunked_docs, embeddings)

        # Sauvegarder le vector store
        if vector_store_path:
            os.makedirs(os.path.dirname(vector_store_path), exist_ok=True)
            vector_store.save_local(vector_store_path)
            logger.info(f"Vector store saved to: {vector_store_path}")

        # Mettre à jour l'état
        state.journal_entries = anonymized_entries
        state.vector_store = vector_store
        state.current_operation_message = (
            f"Journal processed: {len(raw_entries)} entries, "
            f"{len(chunked_docs)} chunks, vector store created."
        )
        state.last_successful_node = "N2_JournalIngestorAnonymizerNode"

        logger.info("N2: Journal ingestion and vectorization complete.")

    except FileNotFoundError:
        error_msg = f"Journal path not found: {journal_path}"
        state.error_message = error_msg
        logger.error(error_msg)
    except Exception as e:
        state.error_message = "Error during journal processing"
        state.error_details = str(e)
        logger.error("Error during journal processing", exc_info=True)

    return state


if __name__ == "__main__":
    # Test du nœud
    test_state = AgentState()
    test_state.journal_path = "data/input/journal_entries"
    test_state.vector_store_path = "data/output/vector_store"

    updated_state = journal_ingestor_anonymizer_node(test_state)

    print("\nUpdated State after N2:")
    if updated_state.error_message:
        print(f"  Error: {updated_state.error_message}")
    else:
        entries_count = (
            len(updated_state.journal_entries) if updated_state.journal_entries else 0
        )
        print(f"  Journal Entries: {entries_count}")
        print(f"  Message: {updated_state.current_operation_message}")
