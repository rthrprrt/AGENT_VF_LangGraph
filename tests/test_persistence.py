# tests/test_persistence.py
import gc
import os
import time
import unittest

# import sqlite3 # Not used in this version
# import pytest # Not used in this version for raises
from langgraph.checkpoint.sqlite import SqliteSaver

from src.config import settings as global_settings  # Renamed to avoid conflict
from src.persistence import get_sqlite_checkpointer


class TestPersistence(unittest.TestCase):
    def setUp(self):
        """Set up for tests."""
        self.original_db_path = global_settings.persistence_db_path
        self.test_db_path = "test_agent_vf_persistence.sqlite"
        global_settings.persistence_db_path = self.test_db_path
        # Ensure no old test DB exists
        if os.path.exists(self.test_db_path):
            # Try to close any existing connections
            gc.collect()
            time.sleep(0.1)  # Give time for connections to close
            try:
                os.remove(self.test_db_path)  # E501 potential
            except PermissionError:
                # If file is still locked, we'll work with it
                pass

    def tearDown(self):
        """Clean up after tests."""
        global_settings.persistence_db_path = self.original_db_path
        # Force garbage collection to close connections
        gc.collect()
        time.sleep(0.1)
        if os.path.exists(self.test_db_path):
            try:
                os.remove(self.test_db_path)
            except PermissionError:  # noqa: S110
                # Ignore errors during cleanup
                pass

    def test_get_sqlite_checkpointer_returns_instance(
        self,
    ):
        """Test that get_sqlite_checkpointer returns a SqliteSaver instance."""
        checkpointer = get_sqlite_checkpointer()
        assert isinstance(checkpointer, SqliteSaver)

    def test_get_sqlite_checkpointer_uses_configured_path(self):
        """
        Test that get_sqlite_checkpointer uses the path from config
        and creates the database file.
        """
        # Call the function, which should create the DB file if it doesn't exist
        # (SqliteSaver.from_conn_string typically does this)
        _ = get_sqlite_checkpointer()
        assert os.path.exists(
            self.test_db_path
        ), f"Database file {self.test_db_path} was not created."

    def test_get_sqlite_checkpointer_raises_error_if_path_is_none(
        self,
    ):
        """Test that get_sqlite_checkpointer raises ValueError if path is None."""
        original_path = global_settings.persistence_db_path
        global_settings.persistence_db_path = None
        with self.assertRaisesRegex(
            ValueError, "Persistence database path .* is not set in settings."
        ):
            get_sqlite_checkpointer()
        global_settings.persistence_db_path = original_path  # Restore

    def test_get_sqlite_checkpointer_raises_error_if_path_is_empty(self):
        """Test that get_sqlite_checkpointer raises ValueError if path is empty."""
        original_path = global_settings.persistence_db_path
        global_settings.persistence_db_path = ""
        with self.assertRaisesRegex(
            ValueError, "Persistence database path .* is not set in settings."
        ):
            get_sqlite_checkpointer()
        global_settings.persistence_db_path = original_path  # Restore


if __name__ == "__main__":
    unittest.main()
