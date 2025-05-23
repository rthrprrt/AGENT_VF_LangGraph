# src/nodes/n2_journal_ingestor_anonymizer.py
import logging
import os
import re
import shutil
import traceback
import uuid
from datetime import date
from typing import Any

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from pydantic import BaseModel, Field, ValidationError

from src.state import AgentState

logger = logging.getLogger(__name__)

DocumentForFAISS = Document
RawJournalEntryDict = dict[str, Any]
ProcessedJournalEntryDict = dict[str, Any]


def _parse_date_from_filename(filename: str) -> date | None:
    """Extracts date from filename if pattern YYYY-MM-DD is found."""
    match = re.search(r"(\d{4}-\d{2}-\d{2})", filename)
    if match:
        try:
            return date.fromisoformat(match.group(1))
        except ValueError:
            # Ligne 33 (E501) - log sur plusieurs lignes
            logger.warning(
                "Format de date invalide dans nom de fichier %s. Date non définie.",
                filename,
            )
    return None


def _load_single_journal_file(
    file_path: str, filename: str
) -> RawJournalEntryDict | None:
    """Loads a single journal file (txt or docx)."""
    # ... (corps de la fonction déjà corrigé pour C901, vérifier les E501 potentiels)
    entry_date_obj = _parse_date_from_filename(filename)
    raw_text_content = ""
    try:
        if filename.lower().endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8")
            docs = loader.load()
            if docs:
                raw_text_content = docs[0].page_content
        elif filename.lower().endswith(".docx"):
            loader = UnstructuredWordDocumentLoader(file_path)
            docs = loader.load()
            if docs:
                raw_text_content = docs[0].page_content
        else:
            logger.debug("Fichier ignoré (extension non supportée): %s", filename)
            return None
        if raw_text_content:
            return {
                "entry_date": entry_date_obj,
                "raw_text": raw_text_content,
                "source_file": filename,
            }
        logger.warning("Aucun contenu extrait de %s", filename)
    except OSError as e:
        logger.error("Erreur I/O chargement fichier %s: %s", filename, e)
    except Exception as e:
        # Ligne 66 (E501) - log sur plusieurs lignes
        logger.error(
            "Erreur générique chargement fichier %s: %s", filename, e, exc_info=True
        )
    return None


def _load_raw_journal_entries(journal_path: str) -> list[RawJournalEntryDict]:
    """Loads raw content from .txt and .docx files in journal_path."""
    entries: list[RawJournalEntryDict] = []
    # ... (corps de la fonction déjà corrigé pour C901)
    if not os.path.isdir(journal_path):
        logger.error(
            "Le chemin du journal %s n'est pas un répertoire valide.", journal_path
        )
        return entries
    for filename in os.listdir(journal_path):
        file_path = os.path.join(journal_path, filename)
        entry = _load_single_journal_file(file_path, filename)
        if entry:
            entries.append(entry)
    logger.info("%d entrées de journal chargées depuis %s.", len(entries), journal_path)
    return entries


def _chunk_entries_for_embedding(
    processed_entries: list[ProcessedJournalEntryDict],
) -> list[DocumentForFAISS]:
    """Chunks processed entries and prepares LangChain Documents for FAISS."""
    # ... (corps de la fonction inchangé, les logs longs sont déjà sur plusieurs lignes)
    logger.info("Démarrage du chunking des entrées traitées...")
    chunk_size = 1500
    chunk_overlap = 200
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    documents_for_faiss: list[DocumentForFAISS] = []
    for entry_idx, entry in enumerate(processed_entries):
        text_to_chunk = entry.get("anonymized_text")
        if not text_to_chunk:
            source_file_info = entry.get("source_file", f"Inconnu_{entry_idx}")
            logger.warning(
                "Aucun 'anonymized_text' pour l'entrée source: %s. Passage.",
                source_file_info,
            )
            continue
        source_file = entry.get("source_file", f"doc_inconnu_{entry_idx}")
        entry_date_obj = entry.get("entry_date")
        journal_date_str = entry_date_obj.isoformat() if entry_date_obj else None
        chunks = text_splitter.split_text(text_to_chunk)
        for i, chunk_text in enumerate(chunks):
            chunk_id = str(uuid.uuid4())
            metadata = {
                "source_document": source_file,
                "journal_date": journal_date_str,
                "chunk_index": i,
                "chunk_id": chunk_id,
            }
            documents_for_faiss.append(
                Document(page_content=chunk_text, metadata=metadata)
            )
    logger.info("%d chunks créés pour l'indexation FAISS.", len(documents_for_faiss))
    return documents_for_faiss


def _save_or_update_faiss_store(
    vector_store_path: str,
    documents_for_faiss: list[DocumentForFAISS],
    embedding_model_name: str,
    recreate_store: bool,
) -> bool:
    """Creates or updates the FAISS vector store."""
    # ... (corps de la fonction inchangé, les logs longs sont déjà sur plusieurs lignes)
    logger.info("Gestion du vector store FAISS à : %s", vector_store_path)
    try:
        embeddings = FastEmbedEmbeddings(model_name=embedding_model_name)
    except Exception as e:
        logger.error(
            "Échec init embedding model (%s): %s",
            embedding_model_name,
            e,
            exc_info=True,
        )
        return False
    vector_store_dir = vector_store_path
    index_file_path = os.path.join(vector_store_dir, "index.faiss")
    if recreate_store and os.path.exists(vector_store_dir):
        logger.info(
            "Recréation. Suppression du vector store existant à : %s", vector_store_dir
        )
        try:
            shutil.rmtree(vector_store_dir)
        except OSError as e:
            logger.error(
                "Erreur OSError suppression vector store %s: %s", vector_store_dir, e
            )
            return False
        except Exception as e:
            logger.error(
                "Erreur suppression vector store %s: %s",
                vector_store_dir,
                e,
                exc_info=True,
            )
            return False
    if not os.path.exists(index_file_path) or recreate_store:
        if not documents_for_faiss:
            logger.warning("Aucun document à indexer, vector store non (re)créé.")
            os.makedirs(vector_store_dir, exist_ok=True)
            return True
        logger.info("Création d'un nouveau vector store FAISS...")
        try:
            os.makedirs(vector_store_dir, exist_ok=True)
            db = FAISS.from_documents(documents_for_faiss, embeddings)
            db.save_local(vector_store_dir)
            logger.info(
                "Vector store FAISS créé et sauvegardé à : %s", vector_store_dir
            )
            return True
        except Exception as e:
            logger.error(
                "Erreur lors de la création du vector store FAISS : %s",
                e,
                exc_info=True,
            )
            return False
    else:
        logger.info(
            "Utilisation du vector store FAISS existant à : %s", vector_store_dir
        )
        return True


class _BaseJournalEntryModel(BaseModel):
    raw_text: str
    source_file: str
    entry_date: date | None = None


class _ProcessedJournalEntryModel(_BaseJournalEntryModel):
    anonymized_text: str
    tone_issues: list[dict[str, str]] = Field(default_factory=list)
    requires_tone_review: bool = False


class _AppSettingsModel:
    llm_model_name: str = "gemma2:9b"


class TextSanitizationExpert:
    """Encapsulates logic for text anonymization and tone management."""

    def __init__(
        self,
        initial_anonymization_map_from_state: dict[str, str],
        app_settings: _AppSettingsModel | None = None,
    ):
        self.settings = app_settings if app_settings else _AppSettingsModel()
        self.anonymization_map = initial_anonymization_map_from_state.copy()
        self.org_chart_map = {
            "Beñat Ortega": "[Le_Directeur_General]",
            "Nicolas Dutreuil": "[Le_Directeur_General_Adjoint_Finances]",
            "Nicolas Broband": "[Le_Directeur_Communication_Financiere]",
            "Thierry Perisser": "[Le_DSI]",
            "TP": "[Le_DSI]",
            "Jérôme Carecchio": "[Mon_Responsable_Projets_Info]",
            "Jérôme": "[Mon_Responsable_Projets_Info]",
            "Romain Hardy": "[Le_Directeur_Corporate_Finance]",
            "Agnès Arnaud": "[L_Assistante_Direction]",
            "Valérie Britay": "[La_Directrice_Adjointe_Pole_Bureau]",
            "Marie Lalande": "[La_Directrice_Executive_Ingenierie_RSE]",
            "Brahim Annour": "[Le_Directeur_Innovation]",
            "Brahim": "[Le_Directeur_Innovation]",
            "Alexandre Morel": "[Le_Chef_Projet_Innovation]",
            "Souhail Ouakkas": "[Le_Stagiaire_IA_Innovation]",
            "Romain Veber": "[Le_Directeur_Executif_Investissements_Developpement]",
            "Pierre-emmanuel Bandioli": "[Le_Directeur_Executif_Pole_Residentiel]",
            "Christine Harné": "[La_Directrice_Executive_RH]",
        }
        for key, value in self.org_chart_map.items():
            if key not in self.anonymization_map:
                self.anonymization_map[key] = value
            elif self.anonymization_map[key] != value and not self.anonymization_map[
                key
            ].startswith("[Person_"):
                # Ligne E501 potentielle, coupée
                log_msg = (
                    f"Clash pour '{key}'. Priorité organigramme: "
                    f"'{value}' vs '{self.anonymization_map[key]}'"
                )
                logger.warning(log_msg)
                self.anonymization_map[key] = value
        self.sorted_org_chart_keys = sorted(
            self.org_chart_map.keys(), key=len, reverse=True
        )
        self.other_companies = {
            "Magic Lemp": "[External_Vendor_A]",
            "Expertime": "[External_Vendor_B]",
        }
        self.whitelisted_projects = ["Héraclès"]
        self.author_name = "Arthur Perrot"
        self.person_counter = sum(
            1 for v in self.anonymization_map.values() if v.startswith("[Person_")
        )
        self.company_counter = sum(
            1
            for v in self.anonymization_map.values()
            if v.startswith("[External_Company_")
        )
        self.project_counter = sum(
            1
            for v in self.anonymization_map.values()
            if v.startswith("[Internal_Project_")
        )
        self.regex_patterns = {
            "project": re.compile(r"\bProjet\s+[A-Za-z0-9\-\_]+\b", re.IGNORECASE),
            "initials": re.compile(r"\b(?:[A-Z]\.){2,}|[A-Z]{2,3}\b"),
        }
        self.tone_keywords_subjective = [
            "je pense que",
            "à mon avis",
            "il me semble",
            "je crois",
            "selon moi",
        ]
        self.tone_keywords_negative_emotion = [
            "frustré",
            "déteste",
            "agacé",
            "terrible",
            "nul",
            "problème",
            "pire",
            "mauvais",
        ]
        # Ligne E501 potentielle (self.tone_keywords_strong_doubt), coupée
        self.tone_keywords_strong_doubt = [
            "ne me rassure pas",
            "pas sûr de",
            "incertain",
            "douteux",
        ]
        self.tone_keywords_informal = ["truc", "machin", "super", "génial", "cool"]
        self.regex_personal_opinion = re.compile(
            r"\b(?:Je|J\')\s+[a-zA-Zà-ÿ']+.*\b", re.IGNORECASE
        )

    def _generate_placeholder(self, entity_type: str) -> str:
        """Generates a new placeholder for a given entity type."""
        if entity_type == "person":
            self.person_counter += 1
            return f"[Person_{chr(ord('A') + self.person_counter - 1)}]"
        elif entity_type == "company":
            self.company_counter += 1
            # Ligne E501 potentielle, coupée
            return f"[External_Company_{chr(ord('A') + self.company_counter - 1)}]"
        elif entity_type == "project":
            self.project_counter += 1
            # Ligne E501 potentielle, coupée
            return f"[Internal_Project_{chr(ord('A') + self.project_counter - 1)}]"
        return "[UNKNOWN_ENTITY]"

    def _apply_specific_replacements(
        self, text: str, replacement_map: dict[str, str], sorted_keys: list[str]
    ) -> str:
        """Applies specific replacements from a map to the text."""
        for key in sorted_keys:
            placeholder = replacement_map[key]
            pattern = r"\b" + re.escape(key) + r"\b"
            if re.search(pattern, text, flags=re.IGNORECASE):
                text = re.sub(pattern, placeholder, text, flags=re.IGNORECASE)
                if key not in self.anonymization_map:
                    self.anonymization_map[key] = placeholder
                    # Ligne E501 potentielle, coupée
                    logger.debug("Entité '%s' remplacée par '%s'", key, placeholder)
        return text

    def _apply_regex_project_replacements(self, text: str) -> str:
        """Applies regex-based project replacements."""
        # ... (Logique déjà OK pour C901)
        project_replacements = []
        for match in self.regex_patterns["project"].finditer(text):
            project_name = match.group(0)
            project_name_lower = project_name.lower()
            is_whitelisted = any(
                wl.lower() == project_name_lower for wl in self.whitelisted_projects
            )
            if not is_whitelisted:
                placeholder = self.anonymization_map.get(project_name)
                if not placeholder:
                    placeholder = self._generate_placeholder("project")
                    self.anonymization_map[project_name] = placeholder
                project_replacements.append((project_name, placeholder))
        project_replacements.sort(key=lambda x: len(x[0]), reverse=True)
        for name, placeholder in project_replacements:
            pattern = r"\b" + re.escape(name) + r"\b"
            text = re.sub(pattern, placeholder, text, flags=re.IGNORECASE)
        return text

    def _anonymize_text_single_entry(self, text: str) -> str:
        """Anonymizes a single text entry."""
        anonymized_text = text
        anonymized_text = self._apply_specific_replacements(
            anonymized_text, self.anonymization_map, self.sorted_org_chart_keys
        )
        anonymized_text = self._apply_specific_replacements(
            anonymized_text,
            self.other_companies,
            sorted(self.other_companies.keys(), key=len, reverse=True),
        )
        anonymized_text = self._apply_regex_project_replacements(anonymized_text)
        return anonymized_text

    def _manage_tone_single_entry(self, text: str) -> tuple[list[dict[str, str]], bool]:
        """Manages tone for a single text entry, flagging issues."""
        tone_issues: list[dict[str, str]] = []
        requires_review = False
        sentences = re.split(r"(?<=[.!?])\s+", text)
        for sentence in sentences:
            if not sentence.strip():
                continue
            flagged = False
            reason = ""
            for keyword_list, reason_template in [
                (self.tone_keywords_subjective, "Subjective_Opinion"),
                (self.tone_keywords_negative_emotion, "Negative_Emotion"),
                (self.tone_keywords_strong_doubt, "Strong_Doubt"),
                (self.tone_keywords_informal, "Informal_Language"),
            ]:
                for keyword in keyword_list:
                    pattern = r"\b" + re.escape(keyword) + r"\b"
                    if re.search(pattern, sentence, re.IGNORECASE):
                        flagged = True
                        reason = reason_template
                        break
                if flagged:
                    break
            if not flagged and self.regex_personal_opinion.search(sentence):
                exclusion_pattern = (
                    r"\b(?:Je|J\')\s+(?:suis|ai|vois|dois|peux|vais|sais|fus"
                    r"|serai|saurai)\b"  # Coupé pour E501
                )
                if not re.search(exclusion_pattern, sentence, re.IGNORECASE):
                    flagged = True
                    reason = "Personal_Statement_Structure"
            if flagged:
                issue_detail = {
                    "original_passage": sentence.strip(),
                    "flag_reason": reason,
                }
                tone_issues.append(issue_detail)
                requires_review = True
                log_passage = sentence.strip()[:50]
                logger.debug(
                    "Problème de ton '%s' marqué: '%.50s...'", reason, log_passage
                )
        return tone_issues, requires_review

    def process_entries(
        self, raw_entries_dicts: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Processes all entries for anonymization and tone."""
        processed_entries_list: list[dict[str, Any]] = []
        for entry_dict in raw_entries_dicts:
            source_file = entry_dict.get("source_file", "Inconnu")
            logger.debug("Traitement de l'entrée: %s", source_file)
            try:
                base_entry = _BaseJournalEntryModel(**entry_dict)
            except ValidationError as e:
                logger.error(
                    "Données d'entrée non conformes pour %s: %s. Passage.",
                    source_file,
                    e,
                )
                failed_entry = entry_dict.copy()
                failed_entry["anonymized_text"] = failed_entry.get("raw_text", "")
                failed_entry["tone_issues"] = [
                    {
                        "original_passage": "ENTRY_VALIDATION_FAILED",
                        "flag_reason": str(e),
                    }
                ]
                failed_entry["requires_tone_review"] = True
                processed_entries_list.append(failed_entry)
                continue

            anonymized_text = self._anonymize_text_single_entry(base_entry.raw_text)
            (tone_issues, requires_tone_review) = self._manage_tone_single_entry(
                anonymized_text
            )
            processed_dict = base_entry.model_dump()
            processed_dict["anonymized_text"] = anonymized_text
            processed_dict["tone_issues"] = tone_issues
            processed_dict["requires_tone_review"] = requires_tone_review
            processed_entries_list.append(processed_dict)
        logger.info(
            "Anonymisation et gestion du ton terminées pour %d entrées.",
            len(processed_entries_list),
        )
        return processed_entries_list


def journal_ingestor_anonymizer_node(state: AgentState) -> dict[str, Any]:
    """Orchestrates ingestion, anonymization, and vectorization of journal entries."""
    # ... (corps de la fonction inchangé, les logs longs sont déjà sur plusieurs lignes)
    logger.info(
        "N2: Démarrage de l'ingestion, anonymisation et vectorisation du journal..."
    )
    updated_fields: dict[str, Any] = {"vector_store_initialized": False}

    if not state.journal_path:
        logger.error("Chemin du journal (journal_path) non défini dans l'état.")
        updated_fields["error_message"] = "Chemin du journal manquant."
        return updated_fields
    if not state.vector_store_path:
        logger.error("Chemin du vector store (vector_store_path) non défini.")
        updated_fields["error_message"] = "Chemin du vector store manquant."
        return updated_fields
    if not state.embedding_model_name:
        logger.error("Modèle d'embedding (embedding_model_name) non défini.")
        updated_fields["error_message"] = "Modèle d'embedding manquant."
        return updated_fields

    try:
        raw_journal_entry_dicts = _load_raw_journal_entries(state.journal_path)
        updated_fields["raw_journal_entries"] = raw_journal_entry_dicts
        if not raw_journal_entry_dicts:
            logger.warning("Aucune entrée de journal brute n'a été chargée.")
        else:
            # Ligne E501 potentielle, coupée
            logger.info("Chargé %d entrées brutes.", len(raw_journal_entry_dicts))

        sanitization_expert = TextSanitizationExpert(state.anonymization_map.copy())
        processed_entry_dicts = sanitization_expert.process_entries(
            raw_journal_entry_dicts
        )
        updated_fields["raw_journal_entries"] = processed_entry_dicts
        updated_fields["anonymization_map"] = sanitization_expert.anonymization_map
        logger.info(
            "Traité %d entrées pour anonymisation/ton.", len(processed_entry_dicts)
        )

        documents_for_faiss = _chunk_entries_for_embedding(processed_entry_dicts)
        if not documents_for_faiss:
            logger.warning("Aucun document généré pour FAISS après chunking.")
        else:
            # Ligne E501 potentielle, coupée
            logger.info("Généré %d documents pour FAISS.", len(documents_for_faiss))

        vector_store_ok = _save_or_update_faiss_store(
            state.vector_store_path,
            documents_for_faiss,
            state.embedding_model_name,
            state.recreate_vector_store,
        )
        updated_fields["vector_store_initialized"] = vector_store_ok

        if vector_store_ok:
            logger.info("Traitement journal et config vector store réussis.")
            updated_fields["current_operation_message"] = (
                "Journal traité et vector store prêt."
            )
        else:
            logger.error("Échec traitement journal ou config vector store.")
            if not updated_fields.get("error_message"):
                updated_fields["error_message"] = "Échec de la config du vector store."
    except Exception as e:
        logger.error("N2: Erreur majeure: %s", e, exc_info=True)
        updated_fields["error_message"] = f"N2 Erreur non gérée: {str(e)}"
        updated_fields["error_details"] = traceback.format_exc()
        updated_fields["vector_store_initialized"] = False

    updated_fields["last_successful_node"] = "N2_JournalIngestorAnonymizerNode"
    return updated_fields
