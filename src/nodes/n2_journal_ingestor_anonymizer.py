# src/nodes/n2_journal_ingestor_anonymizer.py
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

# Langchain document loaders
from langchain_community.document_loaders import (
    TextLoader,
    UnstructuredWordDocumentLoader,  # Pour .docx
)
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.state import AgentState

logger = logging.getLogger(__name__)

DEFAULT_ANONYMIZATION_MAP = {}
DATE_IN_FILENAME_PATTERN = re.compile(
    r"^((\d{4}-\d{2}-\d{2})|(\d{2}[-/]\d{2}[-/]\d{4}))"
)


def simple_anonymizer(text: str, anonymization_map: dict[str, str]) -> str:
    """Applique une anonymisation simple par remplacement de chaînes."""
    for real_name, anon_name in anonymization_map.items():
        text = text.replace(real_name, anon_name)
    return text


def _parse_date_from_filename(filename: str) -> str | None:
    """Extrait une date YYYY-MM-DD d'un nom de fichier si possible."""
    match = DATE_IN_FILENAME_PATTERN.match(filename)
    if match:
        if match.group(2):  # Format YYYY-MM-DD
            return match.group(2)
        if match.group(3):  # Format DD-MM-YYYY ou DD/MM/YYYY
            date_str = match.group(3).replace("/", "-")
            try:
                return datetime.strptime(date_str, "%d-%m-%Y").strftime("%Y-%m-%d")
            except ValueError:  # pragma: no cover
                return date_str  # Retourne la chaîne originale si parsing échoue
    return None


def _load_single_journal_file(file_path: Path) -> list[Document]:
    """
    Charge un unique fichier journal et retourne une liste de Documents Langchain.
    """
    try:
        if file_path.suffix == ".txt":
            loader = TextLoader(str(file_path), encoding="utf-8")
            return loader.load()
        if file_path.suffix == ".docx":
            loader = UnstructuredWordDocumentLoader(str(file_path))
            return loader.load()
        logger.debug("Fichier ignoré (extension non supportée): %s", file_path.name)
        return []
    except Exception as e:  # noqa: BLE001
        logger.error(
            "Erreur générique chargement fichier %s: %s",
            file_path.name,
            e,
            exc_info=True,
        )
        return []


def _load_raw_journal_entries_from_files(journal_dir_path: str) -> list[dict[str, Any]]:
    """Charge toutes les entrées de journal depuis le répertoire spécifié."""
    raw_entries_data: list[dict[str, Any]] = []
    path_obj = Path(journal_dir_path)

    if not path_obj.is_dir():
        logger.error(
            "Le chemin du journal %s n'est pas un répertoire valide.", journal_dir_path
        )
        return raw_entries_data

    for file_path in sorted(path_obj.iterdir()):
        if file_path.is_file():
            docs: list[Document] = []
            try:
                docs = _load_single_journal_file(file_path)
            except Exception as e_load:  # pragma: no cover
                logger.error(
                    "Échec du chargement de _load_single_journal_file pour %s: %s",
                    file_path.name,
                    e_load,
                    exc_info=True,
                )
                continue

            if not docs:
                logger.warning("Aucun document chargé depuis %s", file_path.name)
                continue

            page_content = "\n\n".join(
                [doc.page_content for doc in docs if doc.page_content]
            )

            if page_content.strip():
                date_str = _parse_date_from_filename(file_path.name)
                raw_entries_data.append(
                    {
                        "source_file": file_path.name,
                        "raw_text": page_content,
                        "date_str": date_str if date_str else "Date inconnue",
                        "anonymized_text": page_content,
                        "tone_issues_found": False,
                    }
                )
            else:  # pragma: no cover
                logger.warning(
                    "Aucun contenu textuel extrait de %s après chargement.",
                    file_path.name,
                )

    logger.info(
        "%d entrées de journal chargées depuis %s.",
        len(raw_entries_data),
        journal_dir_path,
    )
    return raw_entries_data


class N2JournalIngestorAnonymizerNode:
    """
    Nœud pour charger, traiter, chunker les entrées du journal et gérer le vector store.

    L'anonymisation est basique. Le vector store utilisé est FAISS.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150):
        """Initialise le nœud avec la taille et le chevauchement des chunks."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _process_entries(
        self, entries: list[dict[str, Any]], anonymization_map: dict[str, str]
    ) -> list[dict[str, Any]]:
        """Applique l'anonymisation aux entrées de journal."""
        processed_entries = []
        for entry in entries:
            entry_copy = entry.copy()
            anon_text = simple_anonymizer(entry_copy["raw_text"], anonymization_map)
            entry_copy["anonymized_text"] = anon_text
            processed_entries.append(entry_copy)
        return processed_entries

    def _chunk_entries_for_embedding(
        self, processed_entries: list[dict[str, Any]]
    ) -> list[Document]:
        """Divise les entrées traitées en chunks pour l'embedding."""
        logger.info("Démarrage du chunking des entrées traitées...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        all_docs: list[Document] = []
        for entry_idx, entry in enumerate(processed_entries):
            anonymized_text = entry.get("anonymized_text")
            if not anonymized_text or not anonymized_text.strip():  # pragma: no cover
                logger.warning(
                    "Aucun 'anonymized_text' pour l'entrée source: %s. Passage.",
                    entry.get("source_file", f"Inconnue_{entry_idx}"),
                )
                continue

            chunks = text_splitter.split_text(anonymized_text)
            for i, chunk_text in enumerate(chunks):
                metadata = {
                    "source_document": entry.get(
                        "source_file", f"Inconnue_{entry_idx}"
                    ),
                    "journal_date": entry.get("date_str", "Date inconnue"),
                    "chunk_index": i,
                    "chunk_id": f"{entry.get('source_file', f'unk_{entry_idx}')}"
                    f"_chunk{i}",
                }
                doc = Document(page_content=chunk_text, metadata=metadata)
                all_docs.append(doc)
        logger.info("%d chunks créés pour l'indexation FAISS.", len(all_docs))
        return all_docs

    def _remove_existing_store(self, vector_store_path: Path) -> None:
        """Supprime un vector store existant."""
        logger.info(
            "Recréation. Suppression du vector store existant à : %s",
            vector_store_path,
        )
        try:
            faiss_file = vector_store_path / "index.faiss"
            pkl_file = vector_store_path / "index.pkl"
            if faiss_file.exists():
                faiss_file.unlink(missing_ok=True)
            if pkl_file.exists():
                pkl_file.unlink(missing_ok=True)

            is_empty_after_unlink = not any(vector_store_path.iterdir())
            if is_empty_after_unlink:  # pragma: no cover
                vector_store_path.rmdir()
            elif not faiss_file.exists() and not pkl_file.exists():
                logger.info(
                    "Fichiers FAISS supprimés, mais le répertoire "
                    "contient d'autres éléments."
                )
            else:  # pragma: no cover
                logger.warning(
                    "Échec de la suppression des fichiers FAISS ou répertoire non vide."
                )
        except Exception as e_del:  # noqa: BLE001 # pragma: no cover
            logger.error("Erreur suppression ancien vector store: %s", e_del)

    def _create_empty_faiss_store(
        self, vector_store_path: Path, embeddings: FastEmbedEmbeddings
    ) -> bool:
        """Crée un vector store FAISS vide."""
        try:
            dummy_doc_for_empty_index = [Document(page_content=" ")]
            empty_faiss = FAISS.from_documents(dummy_doc_for_empty_index, embeddings)
            empty_faiss.save_local(folder_path=str(vector_store_path))
            logger.info("Vector store FAISS vide créé à : %s", vector_store_path)
            return True
        except Exception as e_empty:  # noqa: BLE001 # pragma: no cover
            logger.error("Erreur création vector store vide: %s", e_empty)
            return False

    # noqa: C901
    def _save_or_update_faiss_store(
        self,
        docs_to_index: list[Document],
        vector_store_path_str: str,
        embedding_model_name: str,
        recreate_if_exists: bool,
    ) -> bool:
        """Sauvegarde ou met à jour le vector store FAISS."""
        logger.info("Gestion du vector store FAISS à : %s", vector_store_path_str)
        vector_store_path = Path(vector_store_path_str)
        embeddings: FastEmbedEmbeddings | None = None
        try:
            embeddings = FastEmbedEmbeddings(model_name=embedding_model_name)
        except Exception as e:  # noqa: BLE001
            logger.error(
                "Échec init embedding model (%s): %s",
                embedding_model_name,
                e,
                exc_info=True,
            )
            return False

        if recreate_if_exists and vector_store_path.exists():
            self._remove_existing_store(vector_store_path)

        vector_store_path.mkdir(parents=True, exist_ok=True)

        if not docs_to_index:
            logger.warning(
                "Aucun document à indexer. Vector store non (re)créé ou vide."
            )
            if recreate_if_exists:  # pragma: no cover
                return self._create_empty_faiss_store(vector_store_path, embeddings)
            return True

        try:
            index_file = vector_store_path / "index.faiss"
            pkl_file = vector_store_path / "index.pkl"

            if (
                not recreate_if_exists
                and index_file.exists()
                and pkl_file.exists()
                and index_file.stat().st_size > 0
            ):
                logger.info("Mise à jour du vector store FAISS existant...")
                db = FAISS.load_local(
                    folder_path=str(vector_store_path),
                    embeddings=embeddings,
                    allow_dangerous_deserialization=True,
                )
                db.add_documents(docs_to_index)
            else:
                logger.info("Création d'un nouveau vector store FAISS...")
                db = FAISS.from_documents(docs_to_index, embeddings)

            db.save_local(folder_path=str(vector_store_path))
            store_action = (
                "mis à jour"
                if (not recreate_if_exists and index_file.exists())
                else "créé"
            )
            logger.info(
                "Vector store FAISS %s et sauvegardé à : %s",
                store_action,
                vector_store_path,
            )
            return True
        except Exception as e:  # noqa: BLE001
            logger.error(
                "Erreur lors de la création/mise à jour du vector store FAISS: %s",
                e,
                exc_info=True,
            )
            return False

    def run(self, state: AgentState) -> dict[str, Any]:
        """Exécute le nœud d'ingestion et d'anonymisation du journal."""
        logger.info("N2: Journal Ingestor & Anonymizer Node starting...")
        updated_fields: dict[str, Any] = {}

        if not all(
            [state.journal_path, state.vector_store_path, state.embedding_model_name]
        ):  # pragma: no cover
            msg = (
                "N2 Error: Journal path, vector store path, or embedding model name "
                "missing."
            )
            logger.error(msg)
            updated_fields["error_message"] = msg
            updated_fields["last_successful_node"] = (
                "N2JournalIngestorAnonymizerNode_Error"
            )
            return updated_fields

        raw_journal_data = _load_raw_journal_entries_from_files(state.journal_path)
        if not raw_journal_data:  # pragma: no cover
            logger.warning("N2: Aucune entrée de journal brute n'a été chargée.")

        anonymization_map = state.anonymization_map or DEFAULT_ANONYMIZATION_MAP
        processed_entries = self._process_entries(raw_journal_data, anonymization_map)
        updated_fields["raw_journal_entries"] = processed_entries

        docs_to_index = self._chunk_entries_for_embedding(processed_entries)
        updated_fields["processed_chunks_for_vector_store"] = [
            doc.dict() for doc in docs_to_index
        ]

        store_success = self._save_or_update_faiss_store(
            docs_to_index,
            state.vector_store_path,  # type: ignore
            state.embedding_model_name,  # type: ignore
            state.recreate_vector_store,
        )

        if store_success:
            updated_fields["vector_store_initialized"] = True
            updated_fields["current_operation_message"] = (
                "N2: Journal entries processed and vector store updated."
            )
            updated_fields["last_successful_node"] = "N2JournalIngestorAnonymizerNode"
        else:  # pragma: no cover
            updated_fields["vector_store_initialized"] = False
            updated_fields["error_message"] = (
                "N2: Failed to save or update FAISS vector store."
            )
            updated_fields["last_successful_node"] = (
                "N2JournalIngestorAnonymizerNode_Error"
            )

        logger.info("N2: Journal Ingestor & Anonymizer Node finished.")
        return updated_fields
