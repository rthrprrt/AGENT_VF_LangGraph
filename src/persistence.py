# src/persistence.py
import logging
from pathlib import Path  # Ajout de Path pour la gestion des chemins

from langgraph.checkpoint.sqlite import SqliteSaver

from src.config import settings

logger = logging.getLogger(__name__)


def get_sqlite_checkpointer() -> SqliteSaver:
    """
    Initializes and returns a SqliteSaver instance for LangGraph checkpointing.

    Reads the database path from the global `settings` object.
    Ensures the directory for the SQLite database exists.

    Returns:
        SqliteSaver: An instance of the SQLite checkpointer.

    Raises:
        ValueError: If the persistence_db_path is not set in settings.
        Exception: If SqliteSaver fails to initialize for other reasons.
    """
    db_path_str = settings.persistence_db_path
    if not db_path_str:
        logger.error(
            "Persistence database path is not configured. Cannot initialize "
            "SqliteSaver."
        )
        raise ValueError(
            "Persistence database path (persistence_db_path) is not set in settings."
        )

    # Ensure the directory for the database exists
    db_path = Path(db_path_str)
    db_dir = db_path.parent
    try:
        db_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Ensured directory for SQLite DB exists: %s", db_dir)
    except OSError as e:  # pragma: no cover
        logger.error("Failed to create directory for SQLite DB %s: %s", db_dir, e)
        # On pourrait relever l'erreur ici, mais SqliteSaver pourrait quand même
        # fonctionner si le fichier est créé à la racine, ou échouer proprement.

    logger.info("Initializing SqliteSaver with database path: %s", db_path_str)
    try:
        checkpointer = SqliteSaver.from_conn_string(db_path_str)
        logger.info("SqliteSaver initialized successfully.")
        return checkpointer
    except Exception as e:  # noqa: BLE001
        logger.error(
            "Failed to initialize SqliteSaver with path '%s': %s", db_path_str, e
        )
        raise


if __name__ == "__main__":  # pragma: no cover
    try:
        checkpointer_instance = get_sqlite_checkpointer()
        print(f"Successfully obtained SqliteSaver instance: {checkpointer_instance}")
    except Exception as e:
        print(f"Error in example usage: {e}")
