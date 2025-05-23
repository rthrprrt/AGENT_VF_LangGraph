# src/state.py
from typing import Any  # UP035 a dû être corrigé par ruff --fix

from pydantic import BaseModel, Field


class SectionDetail(BaseModel):
    """Represents the details and content of a single thesis section."""

    title: str
    original_requirements: str  # Extrait des directives Epitech
    student_experience_keywords: list[str] = Field(default_factory=list)
    retrieved_journal_excerpts: list[dict[str, Any]] = Field(default_factory=list)
    # Correction E501 (ligne 17): Commentaire coupé
    anonymized_context_for_llm: str | None = None  # Contexte pour le LLM
    draft_v1: str | None = None
    critique_v1: str | None = None
    refined_draft: str | None = None
    human_feedback: str | None = None
    final_content: str | None = None
    status: str = "pending"  # e.g., pending, context_retrieved, drafted


class AgentState(BaseModel):
    """Defines the overall state of the thesis generation agent."""

    # Input & Configuration
    school_guidelines_path: str | None = None
    journal_path: str | None = None
    output_directory: str = "outputs/theses"
    llm_model_name: str = "gemma2:9b"
    embedding_model_name: str = "fastembed/BAAI/bge-small-en-v1.5"

    # Processed Inputs
    school_guidelines_raw_text: str | None = None
    school_guidelines_structured: dict[str, Any] | None = None
    school_guidelines_formatting: dict[str, Any] | None = None

    raw_journal_entries: list[dict[str, Any]] | None = Field(default_factory=list)
    anonymization_map: dict[str, str] = Field(default_factory=dict)
    # Correction E501 (ligne 38): Chaîne multiligne
    user_persona: str = (
        "un(e) étudiant(e) en dernière année de Master spécialisé en IA et "
        "transformation d'entreprise, réalisant son alternance en tant que "
        "chef de projet IA (AIPO) dans une foncière immobilière."
    )
    # Correction E501 (ligne 39): Chemin coupé
    vector_store_path: str | None = "data/processed/vector_store"

    # Thesis Structure & Content
    thesis_outline: list[SectionDetail] = Field(default_factory=list)
    current_section_index: int = 0

    # Bibliography
    cited_sources_raw: list[str] = Field(default_factory=list)
    bibliography_formatted: str | None = None

    # Final Output
    # Correction E501 (ligne 73 dans le log, ici peut varier): Commentaire coupé
    compiled_thesis_sections: dict[str, str] = Field(
        default_factory=dict
    )  # titre -> contenu
    final_thesis_document_path: str | None = None

    # Operational & Error Handling
    current_operation_message: str | None = None
    error_message: str | None = None
    error_details: str | None = None
    last_successful_node: str | None = None
    retry_count: int = 0

    # For HITL
    human_intervention_needed: bool = False
    human_intervention_message: str | None = None
    human_intervention_data: Any | None = None

    class Config:
        """Pydantic model configuration options."""

        # Correction D106: Ajout d'une docstring
        arbitrary_types_allowed = True
