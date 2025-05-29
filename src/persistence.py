# src/persistence.py
import logging

# import sqlite3 # Was not in the initial version for specific exception handling
from langgraph.checkpoint.sqlite import SqliteSaver

from src.config import settings

logger = logging.getLogger(__name__)


def get_sqlite_checkpointer() -> SqliteSaver:
    """
    Initializes and returns a SqliteSaver instance for LangGraph checkpointing.

    Reads the database path from the global `settings` object.

    Returns:
        SqliteSaver: An instance of the SQLite checkpointer.
    """
    db_path = settings.persistence_db_path
    if not db_path:
        logger.error(
            "Persistence database path is not configured. Cannot initialize "
            "SqliteSaver."
        )
        raise ValueError(
            "Persistence database path (persistence_db_path) is not set in settings."
        )

    logger.info(f"Initializing SqliteSaver with database path: {db_path}")
    try:
        checkpointer = SqliteSaver.from_conn_string(db_path)
        logger.info("SqliteSaver initialized successfully.")
        return checkpointer
    except Exception as e:  # noqa: BLE001
        logger.error(f"Failed to initialize SqliteSaver with path '{db_path}': {e}")
        raise


if __name__ == "__main__":
    # Example usage (for testing or direct script run)
    try:
        checkpointer_instance = get_sqlite_checkpointer()
        print(f"Successfully obtained SqliteSaver instance: {checkpointer_instance}")
        # You might want to test a simple save/load if you had a graph here,
        # but for now, just checking instantiation is enough.
    except Exception as e:  # noqa: BLE001
        print(f"Error in example usage: {e}")
