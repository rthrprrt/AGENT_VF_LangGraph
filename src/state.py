from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field

class SectionDetail(BaseModel):
    title: str
    original_requirements: str # Extrait des directives Epitech
    student_experience_keywords: List[str] = Field(default_factory=list)
    retrieved_journal_excerpts: List[Dict[str, Any]] = Field(default_factory=list)
    anonymized_context_for_llm: Optional[str] = None
    draft_v1: Optional[str] = None
    critique_v1: Optional[str] = None
    refined_draft: Optional[str] = None
    human_feedback: Optional[str] = None
    final_content: Optional[str] = None
    status: str = "pending" # e.g., pending, context_retrieved, drafted, critiqued, human_review_pending, approved

class AgentState(BaseModel):
    # Input & Configuration
    school_guidelines_path: Optional[str] = None
    journal_path: Optional[str] = None # Path to the journal file(s) or directory
    output_directory: str = "outputs/theses"
    llm_model_name: str = "gemma2:9b" # Default, peut être surchargé par config
    embedding_model_name: str = "fastembed/BAAI/bge-small-en-v1.5" # Default FastEmbed model

    # Processed Inputs
    school_guidelines_raw_text: Optional[str] = None
    school_guidelines_structured: Optional[Dict[str, Any]] = None
    school_guidelines_formatting: Optional[Dict[str, Any]] = None

    raw_journal_entries: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    anonymization_map: Dict[str, str] = Field(default_factory=dict)
    user_persona: str = (
        "un(e) étudiant(e) en dernière année de Master spécialisé en IA et transformation d'entreprise, "
        "réalisant son alternance en tant que chef de projet IA (AIPO) dans une foncière immobilière."
    )
    vector_store_path: Optional[str] = "data/processed/vector_store"

    # Thesis Structure & Content
    thesis_outline: List[SectionDetail] = Field(default_factory=list)
    current_section_index: int = 0
    
    # Bibliography
    cited_sources_raw: List[str] = Field(default_factory=list)
    bibliography_formatted: Optional[str] = None

    # Final Output
    compiled_thesis_sections: Dict[str, str] = Field(default_factory=dict) # section_title -> final_content
    final_thesis_document_path: Optional[str] = None
    
    # Operational & Error Handling
    current_operation_message: Optional[str] = None
    error_message: Optional[str] = None
    error_details: Optional[str] = None
    last_successful_node: Optional[str] = None
    retry_count: int = 0

    # Tool specific states (if needed, or managed by tool outputs)
    # e.g., rag_queries_executed: List[str] = Field(default_factory=list)

    # For HITL
    human_intervention_needed: bool = False
    human_intervention_message: Optional[str] = None
    human_intervention_data: Optional[Any] = None # Data to present to human

    class Config:
        arbitrary_types_allowed = True # If you plan to use non-pydantic types like Langchain objects directly in state