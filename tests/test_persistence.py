# tests/test_persistence.py
import gc
import os
import time
import unittest

import pytest  # Ajout pour pytest.raises
from langgraph.checkpoint.sqlite import SqliteSaver

from src.config import settings as global_settings
from src.persistence import get_sqlite_checkpointer


class TestPersistence(unittest.TestCase):
    """Tests pour le module de persistance src/persistence.py."""

    def setUp(self):
        """Configure l'environnement pour chaque test."""
        self.original_db_path = global_settings.persistence_db_path
        self.test_db_path = "test_agent_vf_persistence.sqlite"
        global_settings.persistence_db_path = self.test_db_path
        if os.path.exists(self.test_db_path):
            gc.collect()
            time.sleep(0.1)
            try:
                os.remove(self.test_db_path)
            except PermissionError:  # pragma: no cover
                pass  # Ignorer si le fichier est verrouillé, le test suivant pourrait échouer

    def tearDown(self):
        """Nettoie l'environnement après chaque test."""
        global_settings.persistence_db_path = self.original_db_path
        gc.collect()
        time.sleep(0.1)
        if os.path.exists(self.test_db_path):
            try:
                os.remove(self.test_db_path)
            except PermissionError:  # pragma: no cover
                pass

    def test_get_sqlite_checkpointer_returns_instance(self):
        """Teste que get_sqlite_checkpointer retourne une instance SqliteSaver."""
        checkpointer = get_sqlite_checkpointer()
        assert isinstance(checkpointer, SqliteSaver)

    def test_get_sqlite_checkpointer_uses_configured_path(self):
        """
        Teste que get_sqlite_checkpointer utilise le chemin de config.

        Vérifie également que le fichier de base de données est créé.
        """
        _ = get_sqlite_checkpointer()
        assert os.path.exists(
            self.test_db_path
        ), f"Database file {self.test_db_path} was not created."

    def test_get_sqlite_checkpointer_raises_error_if_path_is_none(self):
        """Teste ValueError si persistence_db_path est None."""
        original_path = global_settings.persistence_db_path
        global_settings.persistence_db_path = None
        with pytest.raises(
            ValueError, match="Persistence database path .* is not set in settings."
        ):
            get_sqlite_checkpointer()
        global_settings.persistence_db_path = original_path

    def test_get_sqlite_checkpointer_raises_error_if_path_is_empty(self):
        """Teste ValueError si persistence_db_path est une chaîne vide."""
        original_path = global_settings.persistence_db_path
        global_settings.persistence_db_path = ""
        with pytest.raises(
            ValueError, match="Persistence database path .* is not set in settings."
        ):
            get_sqlite_checkpointer()
        global_settings.persistence_db_path = original_path


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
