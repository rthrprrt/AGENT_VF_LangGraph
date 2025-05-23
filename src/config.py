# src/config.py
import logging
import os

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

# Configure logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class Settings(BaseModel):
    """Manages application settings."""

    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    llm_model_name: str = os.getenv("LLM_MODEL_NAME", "gemma2:9b")
    embedding_model_name: str = os.getenv(
        "EMBEDDING_MODEL_NAME", "fastembed/BAAI/bge-small-en-v1.5"
    )

    default_school_guidelines_path: str = os.getenv(
        "DEFAULT_SCHOOL_GUIDELINES_PATH",
        "data/input/school_guidelines/Mission_Professionnelle_Digi5_EPITECH.pdf",
    )
    default_journal_path: str = os.getenv(
        "DEFAULT_JOURNAL_PATH", "data/input/journal_entries/"
    )
    default_output_directory: str = os.getenv(
        "DEFAULT_OUTPUT_DIRECTORY", "outputs/theses"
    )
    vector_store_directory: str = os.getenv(
        "VECTOR_STORE_DIRECTORY", "data/processed/vector_store"
    )
    persistence_db_path: str = os.getenv(
        "PERSISTENCE_DB_PATH", "data/processed/langgraph_checkpoints.sqlite"
    )


settings = Settings()
logger.info(
    "Settings loaded: LLM=%s, Embeddings=%s",
    settings.llm_model_name,
    settings.embedding_model_name,
)
