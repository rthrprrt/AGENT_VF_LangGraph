# src/nodes/n3_thesis_outline_planner.py
import logging
import traceback
from typing import Any

from langchain_community.chat_models import ChatOllama
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel as LangchainBaseModel
from langchain_core.pydantic_v1 import Field as LangchainField
from langchain_core.pydantic_v1 import ValidationError as PydanticV1ValidationError

from src.state import AgentState, SectionDetail, SectionStatus

logger = logging.getLogger(__name__)


class PlannedSectionDetailForLLM(LangchainBaseModel):
    """Structure d'une section de thèse que le LLM doit générer."""

    id: str = LangchainField(
        description="Hierarchical ID, e.g., '1.', '1.1.', '2.1.1.', ensuring trailing dot."
    )
    title: str = LangchainField(
        description="Full and exact title of the section/subsection."
    )
    level: int = LangchainField(
        description="Hierarchical level (1 for main part, etc.)."
    )
    description_objectives: str = LangchainField(
        description="What this section should cover, its objectives (2-3 sentences)."
    )
    original_requirements_summary: str = LangchainField(
        description=(
            "Summary of Epitech guidelines for this section (include RNCP if any)."
        )
    )
    student_experience_keywords: list[str] = LangchainField(
        default_factory=list,
        description=("5-10 keywords/phrases from student's journal for N5 to query."),
    )
    example_phrasing_or_content_type: str | None = LangchainField(
        default=None,
        description=("Note on style/content type from example thesis (1-2 sentences)."),
    )
    key_questions_to_answer: list[str] = LangchainField(
        default_factory=list,
        description="2-3 fundamental questions this section must answer.",
    )


class PlannedThesisOutlineForLLM(LangchainBaseModel):
    """Le plan détaillé et structuré pour le mémoire Epitech."""

    outline: list[PlannedSectionDetailForLLM] = LangchainField(
        description=(
            "The list of all PlannedSectionDetailForLLM objects representing the "
            "complete thesis plan, in sequential and hierarchical order."
        )
    )


class N3ThesisOutlinePlannerNode:  # noqa: C901
    """
    Nœud responsable de la génération du plan initial de la thèse.
    """

    def __init__(
        self, llm_model_name: str = "gemma3:12b-it-q4_K_M", temperature: float = 0.05
    ):
        self.llm_model_name = llm_model_name
        self.temperature = temperature
        self.llm: ChatOllama | None = None
        self.structured_llm: Any | None = None
        self.use_fallback_parser: bool = False

        try:
            self.llm = ChatOllama(
                model=self.llm_model_name, temperature=self.temperature, format="json"
            )
            self.structured_llm = self.llm.with_structured_output(
                schema=PlannedThesisOutlineForLLM
            )
            logger.info(
                "N3ThesisOutlinePlannerNode initialized with LLM: %s "
                "using with_structured_output.",
                self.llm_model_name,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "N3 Warning: Initializing with_structured_output for %s failed: %s. "
                "Will use fallback.",
                self.llm_model_name,
                e,
            )
            self.use_fallback_parser = True
            if self.llm is None:
                try:
                    self.llm = ChatOllama(
                        model=self.llm_model_name,
                        temperature=self.temperature,
                        format="json",
                    )
                    logger.info(
                        "N3: LLM for fallback initialized: %s", self.llm_model_name
                    )
                except Exception as init_e:
                    logger.error(
                        "N3 Fatal: ChatOllama(%s) failed to initialize for fallback: %s",
                        self.llm_model_name,
                        init_e,
                        exc_info=True,
                    )

    def _build_prompt_template_str(self) -> str:  # noqa: C901
        """Construit un prompt détaillé pour le LLM."""
        # fmt: off
        persona_block = "**Persona de l'Étudiant :**\n{persona}\n\n"
        guidelines_block = (
            "**Directives Scolaires Epitech (Structurées - `Dict[str, List[str]]`) :**\n"
            "{guidelines_str}\n\n"
        )
        example_block = (
            "**Extrait de l'Exemple de Mémoire (\"Mémoire de Mission Professionnelle – "
            "Digi5\") - Analysez minutieusement sa structure (ex: 1., 1.1, 1.1.1), ses "
            "titres exacts, ses sous-sections, le type de contenu par section (ex: "
            "description, analyse, exemple de projet, réflexion), et comment les "
            "expériences personnelles sont liées aux compétences RNCP :**\n"
            "--- DÉBUT EXEMPLE DE MÉMOIRE ---\n{example_thesis_str}\n"
            "--- FIN EXEMPLE DE MÉMOIRE ---\n\n"
        )
        task_description_block = (
            "**Votre Tâche :**\n"
            "Produisez une liste d'objets `PlannedSectionDetailForLLM` pour composer le "
            "`thesis_outline`. La sortie doit être un objet JSON unique adhérant "
            "strictement au schéma de la classe `PlannedThesisOutlineForLLM` (qui "
            "contient une liste de `PlannedSectionDetailForLLM`).\n\n"
        )
        field_instructions_block = (
            "Pour chaque `PlannedSectionDetailForLLM` :\n"
            "- `id`: Identifiant hiérarchique (ex: \"1.\", \"1.1.\", \"1.1.1.\").\n"
            "- `title`: Titre complet et exact.\n"
            "- `level`: Niveau hiérarchique (1 pour chapitre, 2 pour section, etc.).\n"
            "- `description_objectives`: Objectifs de la section (2-3 phrases).\n"
            "- `original_requirements_summary`: Résumé des directives Epitech pour la "
            "section (inclure RNCP si applicable).\n"
            "- `student_experience_keywords`: Liste de 5-10 mots-clés ou courtes phrases **spécifiques** que l'étudiant (chef de projet IA en foncière immobilière) aurait **probablement utilisés dans son journal d'apprentissage** pour décrire ses tâches, projets, outils, ou réflexions **pertinents pour cette section**. Ces mots-clés doivent être directement utilisables par N5 pour interroger une base vectorielle du journal.\n" # Prompt version before last minor change
            "- `example_phrasing_or_content_type`: (Optionnel) Note sur style/contenu.\n"
            "- `key_questions_to_answer`: 2-3 questions fondamentales pour la section.\n\n"
        )
        standard_sections_block = (
            "**Sections Standards à Inclure (adaptez de l'exemple) :**\n"
            "- \"Avant-propos / Remerciements\"\n"
            "- \"Sommaire Détaillé\"\n"
            "- \"Introduction Générale\"\n"
            "- Chapitres du corps du mémoire (basés sur l'exemple et directives)\n"
            "- \"Conclusion Générale\"\n"
            "- \"Bibliographie\"\n"
            "- \"Annexes\"\n\n"
        )
        final_instruction_block = (
            "Le plan doit être séquentiel et logique. "
            "Répondez UNIQUEMENT avec l'objet JSON `PlannedThesisOutlineForLLM`."
        )

        base_prompt = (
            "Vous êtes un assistant expert en planification académique et en structuration "
            "de documents longs comme des mémoires professionnels. Votre objectif est de "
            "créer un plan de thèse (outline) extrêmement détaillé et pertinent pour un "
            "étudiant d'Epitech Digital School préparant son \"Mémoire de Mission "
            "Professionnelle\" pour le titre RNCP 35284 \"Expert en management des "
            "systèmes d'information\".\n\n"
            "Le plan généré doit être une liste séquentielle d'objets "
            "`PlannedSectionDetailForLLM`, chacun décrivant une section ou sous-section "
            "du mémoire. Vous devez vous baser principalement sur :\n"
            "1.  Les directives officielles de l'école (fournies ci-dessous).\n"
            "2.  Un exemple de mémoire réussi (extrait fourni ci-dessous) qui doit servir "
            "de **modèle principal** pour la structure hiérarchique (y compris les "
            "sous-sous-sections si présentes dans l'exemple pour un sujet donné), la "
            "profondeur, le type de contenu, la nomenclature des titres, et la manière "
            "d'intégrer les expériences personnelles.\n"
            "3.  Le persona de l'étudiant.\n\n"
        )

        prompt_template = (
            base_prompt +
            persona_block +
            guidelines_block +
            example_block +
            task_description_block +
            field_instructions_block +
            standard_sections_block +
            final_instruction_block
        )
        # fmt: on
        return ChatPromptTemplate.from_template(prompt_template)

    def _create_error_section(
        self, id_prefix: str, title_str: str, detail: str
    ) -> SectionDetail:
        """Crée un objet SectionDetail pour signaler une erreur."""
        return SectionDetail(
            id=f"error.N3.{id_prefix}",
            title=title_str,
            level=0,
            description_objectives=detail,
            original_requirements_summary="Erreur de traitement dans N3.",
            student_experience_keywords=["error", id_prefix.lower()],
            key_questions_to_answer=[],
            example_phrasing_or_content_type=None,
            status=SectionStatus.ERROR,
        )

    def run(self, state: AgentState) -> dict[str, Any]:  # noqa: C901
        logger.info("N3: Génération du plan de thèse...")
        updated_fields: dict[str, Any] = {}
        final_thesis_outline: list[SectionDetail] = []

        if not state.school_guidelines_structured or not state.user_persona:
            msg = "N3 Erreur: Directives structurées ou persona manquants."
            logger.error(msg)
            updated_fields["error_message"] = f"N3: {msg}"
            final_thesis_outline.append(
                self._create_error_section("input", "Input Error", msg)
            )
            updated_fields["thesis_outline"] = final_thesis_outline
            updated_fields["last_successful_node"] = state.last_successful_node
            return updated_fields

        example_thesis_text = state.example_thesis_text_content
        if not example_thesis_text:
            logger.warning("N3: Contenu de l'exemple de thèse manquant. Placeholder.")
            example_thesis_text = "Intro, Chapitres, Conclusion."

        guidelines_str_formatted = ""
        if state.school_guidelines_structured:
            for title, points_list in state.school_guidelines_structured.items():
                guidelines_str_formatted += f"\n**{title}**\n"
                if points_list:
                    for point in points_list:
                        guidelines_str_formatted += f"  - {point}\n"
                else:
                    guidelines_str_formatted += (
                        "  - (Aucune sous-directive spécifiée)\n"
                    )
        else:
            guidelines_str_formatted = "Aucune directive scolaire structurée fournie.\n"

        prompt_str = self._build_prompt_template_str().format(
            guidelines_str=guidelines_str_formatted,
            persona=state.user_persona,
            example_thesis_str=example_thesis_text,
        )

        if not self.llm:
            msg = "N3 Erreur Critique: Instance LLM non disponible (échec __init__)."
            logger.error(msg)
            updated_fields["error_message"] = msg
            final_thesis_outline.append(
                self._create_error_section("llm_init", "LLM Critical Error", msg)
            )
            updated_fields["thesis_outline"] = final_thesis_outline
            return updated_fields

        planned_sections_from_llm: list[PlannedSectionDetailForLLM] = []
        raw_json_output_for_debug = "LLM non appelé ou structured_llm a fonctionné."

        try:
            logger.info(
                "N3: Invocation du LLM '%s' pour la planification...",
                self.llm_model_name,
            )
            if not self.use_fallback_parser and self.structured_llm:
                logger.info("N3: Utilisant self.structured_llm.invoke()")
                response_llm_obj: PlannedThesisOutlineForLLM = (
                    self.structured_llm.invoke(prompt_str)
                )
                logger.info("N3: Sortie LLM (with_structured_output) reçue.")
                logger.debug(
                    "N3: Dump JSON sortie LLM (with_structured_output): %s",
                    response_llm_obj.json(indent=2),
                )
                planned_sections_from_llm = response_llm_obj.outline
            else:
                logger.info("N3: Utilisant fallback: invoke().content + parse_raw()")
                llm_response: AIMessage = self.llm.invoke(prompt_str)
                raw_json_output_for_debug = llm_response.content
                logger.debug(
                    "N3: Chaîne JSON brute du LLM: %s", raw_json_output_for_debug
                )
                response_object = PlannedThesisOutlineForLLM.parse_raw(
                    raw_json_output_for_debug
                )
                logger.info("N3: Sortie LLM (fallback parsed) reçue.")
                logger.debug(
                    "N3: Dump JSON sortie LLM (fallback parsed): %s",
                    response_object.json(indent=2),
                )
                planned_sections_from_llm = response_object.outline

            for llm_s_detail in planned_sections_from_llm:
                section_for_state = SectionDetail(
                    id=llm_s_detail.id,
                    title=llm_s_detail.title,
                    level=llm_s_detail.level,
                    description_objectives=llm_s_detail.description_objectives,
                    original_requirements_summary=(
                        llm_s_detail.original_requirements_summary
                    ),
                    student_experience_keywords=(
                        llm_s_detail.student_experience_keywords
                    ),
                    example_phrasing_or_content_type=(
                        llm_s_detail.example_phrasing_or_content_type
                    ),
                    key_questions_to_answer=llm_s_detail.key_questions_to_answer,
                    status=SectionStatus.PENDING,
                )
                final_thesis_outline.append(section_for_state)

            updated_fields["thesis_outline"] = final_thesis_outline
            msg = f"Plan de thèse généré ({len(final_thesis_outline)} sections)."
            updated_fields["current_operation_message"] = msg
            logger.info("N3: %s", msg)

        except PydanticV1ValidationError as ve:
            error_detail_msg = (
                f"N3 Erreur Pydantic: {ve}. "
                f"JSON Brute tenté: {raw_json_output_for_debug[:1000]}..."
            )
            logger.error(error_detail_msg, exc_info=True)
            updated_fields["error_message"] = error_detail_msg
            updated_fields["error_details"] = (
                f"{error_detail_msg}\n{traceback.format_exc()}"
            )
            final_thesis_outline.append(
                self._create_error_section(
                    "pydantic", "LLM Pydantic Error", error_detail_msg
                )
            )
            updated_fields["thesis_outline"] = final_thesis_outline
        except Exception as e:  # noqa: BLE001
            error_detail_msg = f"N3 Erreur générale appel LLM/parsing: {str(e)}"
            logger.error(error_detail_msg, exc_info=True)
            updated_fields["error_message"] = error_detail_msg
            updated_fields["error_details"] = traceback.format_exc()
            final_thesis_outline.append(
                self._create_error_section(
                    "general", "LLM Planning Error", error_detail_msg
                )
            )
            updated_fields["thesis_outline"] = final_thesis_outline

        updated_fields["last_successful_node"] = "N3ThesisOutlinePlannerNode"
        return updated_fields
