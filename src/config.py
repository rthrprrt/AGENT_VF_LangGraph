# src/config.py
import logging
import os

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()


class Settings(BaseModel):
    """Global settings for the AGENT_VF_LangGraph application."""

    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    llm_model_name: str = os.getenv("LLM_MODEL_NAME", "gemma2:9b")
    embedding_model_name: str = os.getenv(
        "EMBEDDING_MODEL_NAME", "fastembed/BAAI/bge-small-en-v1.5"
    )
    # embedding_model_name_st: str = os.getenv(
    #     "EMBEDDING_MODEL_NAME_ST",
    #     "sentence-transformers/all-MiniLM-L6-v2",
    # ) # Alternative

    default_school_guidelines_path: str = (
        "data/input/school_guidelines/Mission_Professionnelle_Digi5_EPITECH.pdf"
    )
    default_journal_path: str = "data/input/journal_entries/"
    default_output_directory: str = "outputs/theses"
    vector_store_directory: str = "data/processed/vector_store"

    log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()

    # LangGraph persistence
    persistence_db_path: str = "data/processed/langgraph_checkpoints.sqlite"


settings = Settings()

# Logging setup
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

logger.info(f"Settings loaded: LLM: {settings.llm_model_name}")
logger.info(f"Embeddings: {settings.embedding_model_name}")
logger.info(f"Log level set to: {settings.log_level}")
