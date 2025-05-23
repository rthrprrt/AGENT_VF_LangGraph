# src/state.py
from datetime import date
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SectionDetail(BaseModel):
    """Details for a single section of the thesis."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    title: str
    original_requirements: str
    student_experience_keywords: list[str] = Field(default_factory=list)
    retrieved_journal_excerpts: list[dict[str, Any]] = Field(default_factory=list)
    anonymized_context_for_llm: str | None = None
    draft_v1: str | None = None
    critique_v1: str | None = None
    refined_draft: str | None = None
    human_feedback: str | None = None
    final_content: str | None = None
    status: str = "pending"


class BaseJournalEntry(BaseModel):
    """Base model for a journal entry, used for parsing/validation."""

    raw_text: str
    source_file: str
    entry_date: date | None = None


class ProcessedJournalEntryState(BaseJournalEntry):
    """Represents a journal entry after anonymization and tone processing."""

    anonymized_text: str
    tone_issues: list[dict[str, str]] = Field(default_factory=list)
    requires_tone_review: bool = False


class AgentState(BaseModel):
    """Represents the overall state of the thesis generation agent."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    school_guidelines_path: str | None = None
    journal_path: str | None = None
    output_directory: str = "outputs/theses"
    llm_model_name: str | None = None
    embedding_model_name: str | None = None
    user_persona: str = (
        "un(e) étudiant(e) en dernière année de Master spécialisé en IA et "
        "transformation d'entreprise, réalisant son alternance en tant que "
        "chef de projet IA (AIPO) dans une foncière immobilière."
    )
    vector_store_path: str | None = None
    recreate_vector_store: bool = False

    school_guidelines_raw_text: str | None = None
    school_guidelines_structured: dict[str, str] | None = None
    school_guidelines_formatting: dict[str, Any] | None = None

    raw_journal_entries: list[dict[str, Any]] = Field(default_factory=list)
    anonymization_map: dict[str, str] = Field(default_factory=dict)

    thesis_outline: list[SectionDetail] = Field(default_factory=list)
    current_section_index: int = 0
    cited_sources_raw: list[str] = Field(default_factory=list)
    bibliography_formatted: str | None = None
    compiled_thesis_sections: dict[str, str] = Field(default_factory=dict)
    final_thesis_document_path: str | None = None
    current_operation_message: str | None = None
    error_message: str | None = None
    # Ligne E501 corrigée :
    error_details: str | None = Field(
        default=None, description="Detailed traceback of an error if any."
    )
    last_successful_node: str | None = None
    vector_store_initialized: bool = False
    human_intervention_needed: bool = False
    human_intervention_message: str | None = None
    human_intervention_data: Any | None = None
