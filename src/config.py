# src/config.py
import logging
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """
    Configuration settings for the AGENT_VF application.

    Values can be overridden by environment variables.
    """

    ollama_base_url: str = "http://localhost:11434"
    llm_model_name: str = "gemma3:12b-it-q4_K_M"
    embedding_model_name: str = "fastembed/BAAI/bge-small-en-v1.5"

    default_school_guidelines_path: str = str(
        PROJECT_ROOT
        / "data/input/school_guidelines/Mission_Professionnelle_Digi5_EPITECH.pdf"
    )
    default_journal_path: str = str(PROJECT_ROOT / "data/input/journal_entries/")
    default_output_directory: str = str(PROJECT_ROOT / "outputs/theses")

    vector_store_directory: str = str(PROJECT_ROOT / "data/processed/vector_store")
    journal_vector_store_path: str = str(PROJECT_ROOT / "data/processed/vector_store")
    recreate_vector_store: bool = False  # Utilisé par N0 et N2

    k_retrieval_count: int = 3

    persistence_db_path: str = str(
        PROJECT_ROOT / "data/processed/langgraph_checkpoints.sqlite"
    )

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
# Le log est maintenant après l'instanciation pour inclure k_retrieval_count
# et persistence_db_path
logger.info(
    "Settings loaded: LLM=%s, Embeddings=%s, RecreateVS=%s, K_Retrieval=%d, "
    "PersistenceDB=%s",
    settings.llm_model_name,
    settings.embedding_model_name,
    settings.recreate_vector_store,
    settings.k_retrieval_count,
    settings.persistence_db_path,
)

if __name__ == "__main__":
    print("Current AGENT_VF Settings (from config.py):")
    # Utiliser model_dump() pour Pydantic V2 / pydantic-settings
    for field_name, value in settings.model_dump().items():
        print(f"  {field_name}: {value}")
