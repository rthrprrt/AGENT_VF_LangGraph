# src/state.py
import logging
from enum import Enum
from typing import Any  # Ajout de List et Optional pour CritiqueOutput

from pydantic.v1 import BaseModel, ConfigDict, Field  # type: ignore

logger = logging.getLogger(__name__)


class SectionStatus(Enum):
    """Possible statuses for a thesis section during generation."""

    PENDING = "PENDING"
    KEYWORDS_GENERATED = "KEYWORDS_GENERATED"
    CONTEXT_RETRIEVED = "CONTEXT_RETRIEVED"
    DRAFT_GENERATED = "DRAFT_GENERATED"
    SELF_CRITIQUE_COMPLETED = "SELF_CRITIQUE_COMPLETED"  # After N7
    HUMAN_REVIEW_PENDING = "HUMAN_REVIEW_PENDING"
    MODIFICATION_REQUESTED = "MODIFICATION_REQUESTED"
    CONTENT_APPROVED = "CONTENT_APPROVED"
    ERROR = "ERROR"
    SKIPPED_BY_USER = "SKIPPED_BY_USER"
    ERROR_CONTEXT_RETRIEVAL = "ERROR_CONTEXT_RETRIEVAL"


class HumanReviewFeedback(BaseModel):
    """Represents feedback from human review for a section."""

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)
    modification_requested: bool = False
    feedback_text: str | None = None


class IdentifiedPoint(BaseModel):
    """Détail d'un point spécifique identifié lors de la critique (pour N7)."""

    point_description: str = Field(
        description="Description claire et concise du point identifié."
    )
    specific_excerpt_from_draft: str | None = Field(
        default=None,
        description="Extrait précis du brouillon concerné, si applicable (max 50 mots).",
    )
    suggested_improvement: str = Field(
        description="Suggestion concrète d'amélioration pour ce point."
    )


class CritiqueOutput(BaseModel):
    """Sortie structurée de la critique d'une section de thèse (pour N7)."""

    overall_assessment_score: int = Field(
        description="Score global qualitatif de la section (de 1=Très faible à 5=Excellent).",
        ge=1,
        le=5,
    )
    overall_assessment_summary: str = Field(
        description="Appréciation générale concise (1-2 phrases) du brouillon."
    )
    identified_flaws: list[IdentifiedPoint] = Field(
        default_factory=list,
        description=(
            "Liste des points faibles spécifiques. Ex: manque de clarté, arguments non "
            "étayés, non-respect d'une directive, mauvaise utilisation du contexte du journal."
        ),
    )
    missing_information: list[IdentifiedPoint] = Field(
        default_factory=list,
        description="Liste d'informations ou d'analyses manquantes pour atteindre les objectifs.",
    )
    superfluous_content: list[IdentifiedPoint] = Field(
        default_factory=list,
        description="Liste de parties du texte redondantes, hors-sujet ou pas assez pertinentes.",
    )
    suggested_search_queries: list[str] = Field(
        default_factory=list,
        description=(
            "Liste de 2-3 mots-clés ou questions de recherche spécifiques et précis pour N5/T1 "
            "si des informations supplémentaires cruciales sont nécessaires et peuvent être trouvées "
            "dans le journal d'apprentissage."
        ),
    )
    final_recommendation: str = Field(
        description=(
            "Recommandation finale : 'REVISION_NEEDED', 'MINOR_EDITS_OK_FOR_REVIEW', ou "
            "'READY_FOR_HUMAN_REVIEW'."
        )
    )


class SectionDetail(BaseModel):
    """Detailed information and content for a single thesis section."""

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
    critique_v1: CritiqueOutput | None = Field(
        default=None, description="Output from N7_SelfCritiqueNode"
    )  # MODIFIÉ
    refined_draft: str | None = None
    human_review_feedback: HumanReviewFeedback | None = None
    final_content: str | None = None
    status: SectionStatus = Field(default=SectionStatus.PENDING)
    error_details_n5_context: str | None = None
    error_details_n6_drafting: str | None = None
    error_details_n7_critique: str | None = None  # Ajouté pour N7
    temporary_human_response: dict[str, Any] | None = Field(
        default=None, description="For N8 HITL response."
    )
    # Pour la boucle Reflexion N6/N7
    current_draft_for_critique: str | None = None
    reflection_history: list[CritiqueOutput] = Field(default_factory=list)
    reflection_attempts: int = 0


class AgentState(BaseModel):
    """
    Represents the overall state of the thesis generation agent.
    """

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

    # Pour l'architecture Plan-and-Execute (future)
    # current_plan: Optional[List[Any]] = None # list of Pydantic models for steps
    # past_steps: List[Tuple[str, str]] = Field(default_factory=list)
    # replan_needed: bool = False

    # Pour la boucle Reflexion (N6/N7) au niveau de la section
    max_reflection_attempts: int = 3  # Configurable

    cited_sources_raw: list[str] = Field(default_factory=list)
    bibliography_formatted: str | None = None
    compiled_thesis_sections: dict[str, str] = Field(default_factory=dict)
    final_thesis_document_path: str | None = None

    last_successful_node: str | None = None
    current_operation_message: str | None = None
    error_message: str | None = None
    error_details: str | None = None
    interrupt_payload: dict[str, Any] | None = Field(
        default=None, description="Payload for HITL interrupt by N8."
    )

    def get_section_by_id(self, section_id: str) -> SectionDetail | None:
        """Retrieves a section from the outline by its ID."""
        for section in self.thesis_outline:
            if section.id == section_id:
                return section
        return None

    def update_section_status(self, section_id: str, new_status: SectionStatus) -> bool:
        """Updates the status of a specific section in the outline."""
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
        logger.warning(  # pragma: no cover
            "Attempted to update status for non-existent section ID: %s", section_id
        )
        return False
