# src/nodes/n0_initial_setup.py
import logging
import os
from typing import Any

from src.config import settings
from src.state import AgentState

logger = logging.getLogger(__name__)


class N0InitialSetupNode:
    """
    Nœud d'initialisation pour configurer les chemins et les modèles de l'agent
    au démarrage.
    """

    def run(self, state: AgentState) -> dict[str, Any]:  # noqa: C901
        """
        Initialise les chemins et configurations des modèles pour l'agent.

        Cette méthode vérifie chaque paramètre de configuration pertinent dans l'état
        de l'agent. Si un paramètre n'est pas défini ou correspond à la valeur
        par défaut du modèle Pydantic, il est initialisé avec la valeur
        provenant des `settings` globaux (configurés via `src/config.py`).
        Les répertoires nécessaires sont également créés si manquants.
        """
        logger.info("N0: Initializing agent settings...")

        current_values_from_state = {
            "school_guidelines_path": state.school_guidelines_path,
            "journal_path": state.journal_path,
            "output_directory": state.output_directory,
            "vector_store_path": state.vector_store_path,
            "llm_model_name": state.llm_model_name,
            "embedding_model_name": state.embedding_model_name,
            "recreate_vector_store": state.recreate_vector_store,
            "user_persona": state.user_persona,
            "example_thesis_text_content": state.example_thesis_text_content,
        }
        # Initialiser updated_fields avec une copie pour s'assurer que les champs non modifiés
        # mais attendus par les tests (comme user_persona si non None) sont présents.
        updated_fields: dict[str, Any] = current_values_from_state.copy()

        # School Guidelines Path
        if current_values_from_state["school_guidelines_path"] is None:
            default_path = settings.default_school_guidelines_path
            updated_fields["school_guidelines_path"] = default_path
            logger.info(
                "  Set school_guidelines_path to default: %s",
                default_path,
            )

        # Journal Path
        if current_values_from_state["journal_path"] is None:
            default_path = settings.default_journal_path
            updated_fields["journal_path"] = default_path
            logger.info("  Set journal_path to default: %s", default_path)

        # Output Directory
        current_output_dir = updated_fields[
            "output_directory"
        ]  # Lire depuis updated_fields (qui a la copie de state)
        output_dir_field = AgentState.__fields__.get("output_directory")
        if output_dir_field and (
            current_output_dir is None or current_output_dir == output_dir_field.default
        ):
            default_path = settings.default_output_directory
            updated_fields["output_directory"] = default_path
            current_output_dir = default_path
            logger.info("  Set output_directory to default: %s", current_output_dir)

        if current_output_dir:
            os.makedirs(current_output_dir, exist_ok=True)
            logger.info("  Ensured output directory exists: %s", current_output_dir)
        else:
            logger.error(
                "N0: Output directory is not set and no default could be applied."
            )
            updated_fields["error_message"] = "N0: Output directory missing"
            if (
                "output_directory" not in updated_fields
            ):  # S'assurer qu'il est là pour les tests
                updated_fields["output_directory"] = None

        # Vector Store Path
        current_vector_store_path = updated_fields["vector_store_path"]
        vector_store_path_field = AgentState.__fields__.get("vector_store_path")

        path_is_none_or_default = False
        if vector_store_path_field:
            if (
                current_vector_store_path is None
                or current_vector_store_path == vector_store_path_field.default
            ):
                path_is_none_or_default = True
        elif current_vector_store_path is None:
            path_is_none_or_default = True

        if path_is_none_or_default:
            default_vs_path = settings.vector_store_directory
            updated_fields["vector_store_path"] = default_vs_path
            current_vector_store_path = default_vs_path
            logger.info(
                "  Set vector_store_path to default from settings: %s",
                current_vector_store_path,
            )

        if current_vector_store_path:
            os.makedirs(current_vector_store_path, exist_ok=True)
            logger.info(
                "  Ensured vector store directory exists: %s",
                current_vector_store_path,
            )
        else:
            logger.error(
                "N0: Vector store path is not set and no default could be applied."
            )
            updated_fields["error_message"] = (
                f'{updated_fields.get("error_message", "")} N0: Vector store path missing'.strip()
            )
            if (
                "vector_store_path" not in updated_fields
            ):  # S'assurer qu'il est là pour les tests
                updated_fields["vector_store_path"] = None

        # LLM and Embedding models
        if current_values_from_state["llm_model_name"] is None:
            updated_fields["llm_model_name"] = settings.llm_model_name
            logger.info("  Set LLM model to: %s", settings.llm_model_name)

        if current_values_from_state["embedding_model_name"] is None:
            updated_fields["embedding_model_name"] = settings.embedding_model_name
            logger.info(
                "  Set embedding model to: %s",
                settings.embedding_model_name,
            )

        state_dict_set_fields = state.dict(exclude_unset=True)
        if "recreate_vector_store" not in state_dict_set_fields:
            updated_fields["recreate_vector_store"] = settings.recreate_vector_store
            logger.info(
                "  Set recreate_vector_store from settings: %s",
                settings.recreate_vector_store,
            )

        user_persona_field = AgentState.__fields__.get("user_persona")
        current_user_persona = updated_fields[
            "user_persona"
        ]  # Lire depuis la copie de l'état

        if (
            user_persona_field
            and (
                current_user_persona is None
                or current_user_persona == user_persona_field.default
            )
            and user_persona_field.default is not None
        ):
            # Si le persona est None ou déjà le défaut Pydantic, il n'est pas nécessaire de le mettre à jour
            # updated_fields["user_persona"] = user_persona_field.default # Déjà fait par la copie initiale
            logger.info("  User persona is Pydantic default from AgentState model.")
        elif current_user_persona is not None:
            # Déjà dans updated_fields depuis la copie initiale
            logger.info(
                "  User persona preserved from initial state: %s", current_user_persona
            )
        else:
            fallback_persona = "Generic Student Persona (N0 Fallback)"
            updated_fields["user_persona"] = fallback_persona
            logger.warning(
                "  User persona was None and no Pydantic default, set a hardcoded N0 default: %s",
                fallback_persona,
            )

        if updated_fields["example_thesis_text_content"] is None:
            logger.info(
                "  example_thesis_text_content is None, no default set in N0 from settings."
            )

        updated_fields["last_successful_node"] = "N0InitialSetupNode"
        logger.info("N0: Agent settings initialization complete.")
        return updated_fields
