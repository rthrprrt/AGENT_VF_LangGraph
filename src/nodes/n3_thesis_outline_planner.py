# src/nodes/n3_thesis_outline_planner.py
import logging
import traceback
import json # Ajout pour la Solution 2 si nécessaire plus tard
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
        description="Hierarchical ID, e.g., '1.', '1.1.', '2.1.1.', "
        "ensuring trailing dot."
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
        """Initialise le nœud avec le modèle LLM et la température."""
        self.llm_model_name = llm_model_name
        self.temperature = temperature
        self.llm: ChatOllama | None = None
        self.structured_llm: Any | None = None
        self.use_fallback_parser: bool = False

        try:
            self.llm = ChatOllama(
                model=self.llm_model_name, temperature=self.temperature, format="json"
            )
            # Tenter with_structured_output, mais se préparer au fallback
            self.structured_llm = self.llm.with_structured_output(
                schema=PlannedThesisOutlineForLLM,
                # Certains modèles nécessitent que le nom du schéma soit dans le prompt
                # name="PlannedThesisOutlineForLLM" 
            )
            logger.info(
                "N3ThesisOutlinePlannerNode initialized with LLM: %s "
                "using with_structured_output.",
                self.llm_model_name,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "N3 Warning: Initializing with_structured_output for %s failed. "
                "Error: %s. Traceback: %s. Will use fallback.",
                self.llm_model_name,
                e,
                traceback.format_exc()
            )
            self.use_fallback_parser = True
            # S'assurer que self.llm est initialisé même si structured_llm échoue
            if self.llm is None:  # pragma: no cover
                try:
                    self.llm = ChatOllama(
                        model=self.llm_model_name,
                        temperature=self.temperature,
                        format="json", # Demander explicitement du JSON au LLM
                    )
                    logger.info(
                        "N3: LLM for fallback initialized: %s", self.llm_model_name
                    )
                except Exception as init_e:  # pragma: no cover
                    logger.error(
                        "N3 Fatal: ChatOllama(%s) failed to initialize for fallback: %s",
                        self.llm_model_name,
                        init_e,
                        exc_info=True,
                    )

    def _build_prompt_template_str(self) -> str:  # noqa: C901
        """Construit un prompt détaillé pour le LLM."""
        # fmt: off
        persona_block = """**Persona de l'Étudiant (Contexte pour la génération des keywords) :**
{persona}
"""
        guidelines_block = (
            "**Directives Scolaires Epitech (Structurées - `Dict[str, List[str]]`) - Votre source d'autorité pour les exigences de contenu et de structure de chaque section :**\n"
            "{guidelines_str}\n\n"
        )
        example_block = (
            "**Exemple de Mémoire de Référence COMPLET (\"Mémoire de Mission Professionnelle – Digi5\") - Votre **MODÈLE PRINCIPAL ABSOLU** pour la structure hiérarchique, la nomenclature des titres, la profondeur, l'ordre des sections, et le type de contenu attendu par section. Analysez-le avec une extrême minutie pour reproduire une architecture de plan similaire et pertinente pour le sujet de l'étudiant actuel. Si Digi5 a des sous-sous-sections (ex: 1.1.1, 1.1.2), votre plan DOIT refléter une granularité similaire pour les parties équivalentes. L'intégration des expériences de l'étudiant doit s'inspirer de la manière dont Digi5 illustre les compétences RNCP avec des exemples concrets.**\n"
            "--- DÉBUT EXEMPLE DE MÉMOIRE COMPLET ---\n{example_thesis_str}\n"
            "--- FIN EXEMPLE DE MÉMOIRE COMPLET ---\n\n"
        )
        task_description_block = (
            "**Votre Tâche Fondamentale :**\n"
            "En tant qu'expert en planification académique, votre unique objectif est de produire un **plan de thèse (outline) COMPLET, EXTRÊMEMENT DÉTAILLÉ, et PARFAITEMENT STRUCTURÉ** pour un étudiant d'Epitech Digital School préparant son \"Mémoire de Mission Professionnelle\" (titre RNCP 35284 \"Expert en management des systèmes d'information\").\n"
            "La sortie doit être un objet JSON unique. Cet objet JSON doit avoir une **unique clé racine nommée `outline`**. La valeur de cette clé `outline` doit être une liste d'objets, où chaque objet adhère **strictement** au schéma Pydantic `PlannedSectionDetailForLLM`.\n\n" # MODIFIÉ ICI
        )
        field_instructions_block = (
            "**Instructions Détaillées pour Chaque Objet `PlannedSectionDetailForLLM` (Soyez Méticuleux) :**\n"
            "- `id`: str - Identifiant hiérarchique exact et unique, avec point final (ex: \"1.\", \"1.1.\", \"1.1.1.\"). Doit refléter la structure de l'exemple Digi5 et les directives Epitech. La numérotation doit être continue et logique.\n"
            "- `title`: str - Titre complet, précis et académique de la section/sous-section. Inspirez-vous fortement de la nomenclature et du style des titres de l'exemple Digi5, tout en l'adaptant au sujet spécifique de l'étudiant.\n"
            "- `level`: int - Niveau hiérarchique (1 pour une partie/chapitre principal, 2 pour une section majeure, 3 pour une sous-section, etc.), en stricte conformité avec la profondeur observée dans l'exemple Digi5.\n"
            "- `description_objectives`: str - Description concise (2-4 phrases claires et distinctes) des objectifs principaux de cette section. Que doit-elle démontrer ou expliquer ? Quel est son rôle dans l'argumentation globale du mémoire ?\n"
            "- `original_requirements_summary`: str - Un résumé ciblé des directives Epitech spécifiquement applicables à cette section (ou à la partie du mémoire qu'elle représente). Mentionnez explicitement les compétences RNCP si les directives les associent à ce type de contenu.\n"
            "- `student_experience_keywords`: List[str] - **POINT ABSOLUMENT CRUCIAL.** Générez une liste de 5 à 10 mots-clés ou courtes phrases (2-5 mots maximum par phrase-clé) **extrêmement spécifiques, concrets, et directement exploitables pour une recherche RAG dans le journal d'apprentissage de l'étudiant.** Ces keywords doivent être des termes que l'étudiant (chef de projet IA en foncière immobilière, travaillant sur des projets comme 'Automatisation DG', 'Prompt RH', 'Power BI Finance', utilisant 'Power Automate', 'AI Builder', 'Azure Functions', 'Python', 'Microsoft Copilot Studio') aurait **probablement utilisés dans son journal pour décrire ses tâches, projets, outils, réflexions, ou problèmes rencontrés, et qui sont DIRECTEMENT PERTINENTS pour alimenter CETTE SECTION PRÉCISE du mémoire.** Évitez toute généricité. Pensez \"termes de recherche Google ultra-ciblés pour le journal de cet étudiant\".\n"
            "    *   Exemples de BONS keywords (pour une section sur l'analyse des besoins du 'Projet Automatisation DG'): ['analyse besoins Direction Générale', 'cartographie processus DG', 'workflow DG pain points', 'specs fonctionnelles Power Automate DG', 'exigences métier automatisation DG', 'ateliers utilisateurs DG', 'compte-rendu COPROJ DG automatisation']\n"
            "    *   Exemples de MAUVAIS keywords (trop vagues): ['analyse', 'besoins', 'projet IA', 'automatisation', 'utilisateurs', 'gestion de projet']\n"
            "- `example_phrasing_or_content_type`: Optional[str] - (Fortement recommandé) Une note brève (1-2 phrases) sur le style de rédaction, le type de contenu attendu, ou une formulation exemplaire pour cette section, en se basant sur une section équivalente dans l'exemple Digi5. Exemples: ['Analyse critique des résultats du projet X en utilisant la méthode STAR.', 'Présentation des solutions techniques envisagées sous forme de tableau comparatif.', 'Réflexion personnelle sur les défis de gestion du changement rencontrés.']\n"
            "- `key_questions_to_answer`: List[str] - Une liste de 2 à 4 questions fondamentales et précises auxquelles cette section doit impérativement apporter une réponse claire et argumentée pour atteindre ses objectifs.\n\n"
        )
        standard_sections_block = (
            "**Structure Générale Attendue (Adaptez les titres exacts et la hiérarchie de l'exemple Digi5 et des directives Epitech) :**\n"
            "Le plan doit couvrir, au minimum, les grandes phases d'un mémoire Epitech, telles qu'illustrées par Digi5 :\n"
            "- Sections préliminaires (Avant-propos/Remerciements, Sommaire Détaillé -- ceux-ci peuvent avoir des `student_experience_keywords` plus génériques ou liés à la méthodologie de rédaction du mémoire lui-même).\n"
            "- Introduction Générale (avec ses propres sous-sections si Digi5 en a : Contexte, Problématique, Objectifs du mémoire, Annonce du plan).\n"
            "- Corps du Mémoire : Typiquement 2 à 4 chapitres/parties principaux, chacun décomposé en sections et sous-sections. **C'est ici que la fidélité à la structure et à la granularité de Digi5 est la plus critique.** Un chapitre doit être dédié à l'analyse des compétences RNCP, illustré par les expériences du journal.\n"
            "- Conclusion Générale (avec ses propres sous-sections si Digi5 en a : Synthèse, Limites, Ouvertures).\n"
            "- Sections finales (Bibliographie, Annexes -- pour ces sections, les `student_experience_keywords` peuvent être non applicables ou spécifiques à la recherche bibliographique).\n\n"
        )
        final_instruction_block = (
            "**Impératifs Finaux :**\n"
            "1.  Le plan généré doit être **exhaustif** et couvrir l'intégralité du mémoire, du début à la fin.\n"
            "2.  La **séquence et la hiérarchie** des sections doivent être logiques et suivre scrupuleusement le modèle de Digi5 et les directives Epitech.\n"
            "3.  Soyez **extrêmement précis et détaillé** pour chaque champ de chaque `PlannedSectionDetailForLLM`.\n"
            "4.  Votre unique sortie doit être un **objet JSON unique et valide**. Cet objet JSON doit avoir une **SEULE clé racine nommée `outline`**. La valeur de cette clé `outline` doit être la liste des objets `PlannedSectionDetailForLLM`." # MODIFIÉ ICI
        )

        base_prompt = (
            "Vous êtes un assistant expert en ingénierie pédagogique et en structuration de mémoires académiques de niveau Master. Votre mission est de créer un plan de thèse (outline) exceptionnellement détaillé, pertinent et parfaitement structuré pour un étudiant d'Epitech Digital School (titre RNCP 35284 \"Expert en management des systèmes d'information\").\n\n"
            "Le plan doit être une liste séquentielle d'objets `PlannedSectionDetailForLLM`. Vous devez vous baser sur TROIS sources d'information primordiales :\n"
            "1.  Le persona de l'étudiant (pour guider la pertinence des `student_experience_keywords`).\n"
            "2.  Les directives officielles de l'école (pour les exigences de contenu et de structure).\n"
            "3.  Un exemple COMPLET de mémoire réussi (Digi5), qui est votre **référence principale absolue** pour la structure hiérarchique, la nomenclature des titres, la profondeur des sections, et l'intégration des expériences.\n\n"
        )

        prompt_template_str = (
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
        return ChatPromptTemplate.from_template(prompt_template_str)

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
        """Exécute la génération du plan de thèse."""
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
        if not example_thesis_text:  # pragma: no cover
            logger.warning("N3: Contenu de l'exemple de thèse manquant. Placeholder.")
            example_thesis_text = "Intro, Chapitres, Conclusion."

        guidelines_str_formatted = ""
        if state.school_guidelines_structured:
            for title, points_list in state.school_guidelines_structured.items():
                guidelines_str_formatted += f"\n**{title}**\n"
                if points_list:
                    for point in points_list:
                        guidelines_str_formatted += f"  - {point}\n"
                else:  # pragma: no cover
                    guidelines_str_formatted += (
                        "  - (Aucune sous-directive spécifiée)\n"
                    )
        else:  # pragma: no cover
            guidelines_str_formatted = "Aucune directive scolaire structurée fournie.\n"

        prompt_input_for_llm = self._build_prompt_template_str().format(
            guidelines_str=guidelines_str_formatted,
            persona=state.user_persona,
            example_thesis_str=example_thesis_text,
        )

        if not self.llm:  # pragma: no cover
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
            if not self.use_fallback_parser and self.structured_llm: # pragma: no cover (car on sait qu'il échoue)
                logger.info("N3: Utilisant self.structured_llm.invoke()")
                response_llm_obj: PlannedThesisOutlineForLLM = (
                    self.structured_llm.invoke(prompt_input_for_llm)
                )
                logger.info("N3: Sortie LLM (with_structured_output) reçue.")
                logger.debug(
                    "N3: Dump JSON sortie LLM (with_structured_output): %s",
                    response_llm_obj.json(indent=2),
                )
                planned_sections_from_llm = response_llm_obj.outline
            else:
                logger.info("N3: Utilisant fallback: invoke().content + parse_raw()")
                llm_response: AIMessage = self.llm.invoke(prompt_input_for_llm)
                raw_json_output_for_debug = llm_response.content
                logger.info("N3 RAW LLM OUTPUT (FALLBACK):\n%s", raw_json_output_for_debug)
                
                # Tentative de correction du JSON si la clé racine est 'planned_thesis_outline'
                corrected_json_str = raw_json_output_for_debug
                try:
                    parsed_candidate = json.loads(raw_json_output_for_debug)
                    if "planned_thesis_outline" in parsed_candidate and "outline" not in parsed_candidate:
                        logger.warning("N3: Tentative de correction de la clé racine du JSON de 'planned_thesis_outline' vers 'outline'.")
                        parsed_candidate["outline"] = parsed_candidate.pop("planned_thesis_outline")
                        corrected_json_str = json.dumps(parsed_candidate)
                except json.JSONDecodeError:
                    logger.error("N3: Le JSON brut du LLM n'est pas un JSON valide et ne peut être corrigé.")
                    # L'erreur Pydantic sera levée par parse_raw ci-dessous

                try:
                    response_object = PlannedThesisOutlineForLLM.parse_raw(
                        corrected_json_str # Utiliser la chaîne potentiellement corrigée
                    )
                    logger.info("N3: Fallback parsing successful.")
                except PydanticV1ValidationError as ve_fallback: 
                    logger.error("N3 FALLBACK PARSING FAILED: %s", ve_fallback)
                    logger.error("N3 JSON that failed parsing (après correction éventuelle):\n%s", corrected_json_str)
                    raise ve_fallback 

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
            error_detail_msg_parts = [
                f"N3 Erreur Pydantic: {ve}.",
                f"JSON Brute tenté (ou corrigé): {raw_json_output_for_debug[:500]}...",
            ]
            error_detail_msg = " ".join(error_detail_msg_parts)

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