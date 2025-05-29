# src/state.py
import logging
from enum import Enum
from typing import Any

from pydantic.v1 import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class SectionStatus(Enum):
    PENDING = "PENDING"
    KEYWORDS_GENERATED = "KEYWORDS_GENERATED"
    CONTEXT_RETRIEVED = "CONTEXT_RETRIEVED"
    DRAFT_GENERATED = "DRAFT_GENERATED"
    SELF_CRITIQUE_COMPLETED = "SELF_CRITIQUE_COMPLETED"
    HUMAN_REVIEW_PENDING = "HUMAN_REVIEW_PENDING"
    MODIFICATION_REQUESTED = "MODIFICATION_REQUESTED"
    CONTENT_APPROVED = "CONTENT_APPROVED"
    ERROR = "ERROR"
    SKIPPED_BY_USER = "SKIPPED_BY_USER"


class HumanReviewFeedback(BaseModel):
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)
    modification_requested: bool = False
    feedback_text: str | None = None


class SectionDetail(BaseModel):
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)
    id: str = Field(description="Hierarchical ID, e.g., '1.', '1.1.', '2.1.1.'")
    title: str = Field(description="Full title of the section/subsection.")
    level: int = Field(description="Hierarchical level (1 for main part, etc.).")
    description_objectives: str = Field(description="What this section should cover.")
    original_requirements_summary: str = Field(
        description="Summary of Epitech guidelines."
    )
    student_experience_keywords: list[str] = Field(
        default_factory=list, description="Keywords from journal for RAG."
    )
    example_phrasing_or_content_type: str | None = Field(
        default=None, description="Note on style/content from example."
    )
    key_questions_to_answer: list[str] = Field(
        default_factory=list, description="Questions section must answer."
    )
    retrieved_journal_excerpts: list[dict[str, Any]] = Field(default_factory=list)
    anonymized_context_for_llm: str | None = None
    draft_v1: str | None = None
    critique_v1: str | None = None
    refined_draft: str | None = None
    human_review_feedback: HumanReviewFeedback | None = None
    final_content: str | None = None
    status: SectionStatus = Field(default=SectionStatus.PENDING)
    error_details_n5_context: str | None = None
    error_details_n6_drafting: str | None = None
    temporary_human_response: dict[str, Any] | None = Field(
        default=None, description="For N8 HITL response."
    )


class AgentState(BaseModel):
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)
    school_guidelines_path: str | None = None
    journal_path: str | None = None
    output_directory: str | None = None
    vector_store_path: str | None = None
    llm_model_name: str | None = None
    embedding_model_name: str | None = None
    recreate_vector_store: bool = False
    user_persona: str = (
        "un(e) étudiant(e) en dernière année de Master spécialisé en IA et "
        "transformation d'entreprise, réalisant son alternance en tant que "
        "chef de projet IA (AIPO) dans une foncière immobilière."
    )
    example_thesis_text_content: str | None = None
    school_guidelines_raw_text: str | None = None
    school_guidelines_structured: dict[str, list[str]] | None = Field(
        default_factory=dict
    )
    school_guidelines_formatting: dict[str, Any] = Field(default_factory=dict)
    raw_journal_entries: list[dict[str, Any]] = Field(default_factory=list)
    anonymization_map: dict[str, str] = Field(default_factory=dict)
    vector_store_initialized: bool = False
    processed_chunks_for_vector_store: list[dict[str, Any]] | None = None
    thesis_outline: list[SectionDetail] = Field(default_factory=list)
    current_section_id: str | None = None
    current_section_index: int = 0
    current_section_index_for_router: int = 0
    cited_sources_raw: list[str] = Field(default_factory=list)
    bibliography_formatted: str | None = None
    compiled_thesis_sections: dict[str, str] = Field(default_factory=dict)
    final_thesis_document_path: str | None = None
    last_successful_node: str | None = None
    current_operation_message: str | None = None
    error_message: str | None = None
    error_details: str | None = None
    interrupt_payload: dict[str, Any] | None = Field(
        default=None, description="Payload for HITL interrupt."
    )

    def get_section_by_id(self, section_id: str) -> SectionDetail | None:
        for section in self.thesis_outline:
            if section.id == section_id:
                return section
        return None

    def update_section_status(self, section_id: str, new_status: SectionStatus) -> bool:
        section = self.get_section_by_id(section_id)
        if section:
            section.status = new_status
            logger.info(
                "Status of section '%s' (ID: %s) updated to %s.",
                section.title,
                section_id,
                new_status.value,
            )
            return True
        logger.warning(
            "Attempted to update status for non-existent section ID: %s", section_id
        )
        return False
