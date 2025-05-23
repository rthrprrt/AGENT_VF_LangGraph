# src/state.py
from typing import Any

from pydantic import BaseModel, Field, ConfigDict


class SectionDetail(BaseModel):
    """Details for a single section of the thesis.
    
    This model represents the detailed state and content for each section
    of the thesis being generated, tracking everything from initial
    requirements to final content.
    """
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    title: str
    original_requirements: str
    student_experience_keywords: list[str] = Field(default_factory=list)
    retrieved_journal_excerpts: list[dict[str, Any]] = Field(
        default_factory=list
    )
    anonymized_context_for_llm: str | None = None
    draft_v1: str | None = None
    critique_v1: str | None = None
    refined_draft: str | None = None
    human_feedback: str | None = None
    final_content: str | None = None
    status: str = "pending"


class AgentState(BaseModel):
    """Represents the overall state of the thesis generation agent.
    
    This model contains all the state information needed for the thesis
    generation process, including input configuration, processed data,
    thesis structure, and operational status.
    """
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # --- Input & Configuration ---
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

    # --- Processed Inputs ---
    school_guidelines_raw_text: str | None = None
    school_guidelines_structured: dict[str, str] | None = None
    school_guidelines_formatting: dict[str, Any] | None = None

    journal_entries: list[dict[str, Any]] = Field(default_factory=list)
    raw_journal_entries: list[dict[str, Any]] = Field(default_factory=list)
    anonymization_map: dict[str, str] = Field(default_factory=dict)

    # --- Thesis Structure & Content ---
    thesis_outline: list[SectionDetail] = Field(default_factory=list)
    current_section_index: int = 0

    # --- Bibliography ---
    cited_sources_raw: list[str] = Field(default_factory=list)
    bibliography_formatted: str | None = None

    # --- Final Output ---
    compiled_thesis_sections: dict[str, str] = Field(default_factory=dict)
    final_thesis_document_path: str | None = None

    # --- Operational & Error Handling ---
    current_operation_message: str | None = None
    error_message: str | None = None
    error_details: str | None = None
    last_successful_node: str | None = None

    # --- RAG Specific State ---
    vector_store: Any | None = None
    vector_store_initialized: bool = False

    # --- HITL Specific State ---
    human_intervention_needed: bool = False
    human_intervention_message: str | None = None
    human_intervention_data: Any | None = None