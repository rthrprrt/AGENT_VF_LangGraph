# src/nodes/n6_section_drafting.py
import logging
from typing import Any

from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from src.config import settings  # Pour le nom du modèle LLM, etc.

# Si vous utilisez des modèles Pydantic pour les IO du LLM
# from langchain_core.output_parsers import StrOutputParser # Si la sortie est une simple chaîne
from src.state import AgentState, SectionDetail, SectionStatus

logger = logging.getLogger(__name__)

# Optionnel: Définir un modèle Pydantic pour la sortie attendue si elle est structurée
# class SectionDraftOutput(BaseModel):
#     draft_text: str = Field(description="The drafted content of the section.")
#     any_issues_found: Optional[str] = Field(default=None, description="Any issues noted by LLM during drafting.")


class N6SectionDraftingNode:
    """
    Node responsible for drafting the content of a specific thesis section
    based on the plan, retrieved context, and overall guidelines.
    """

    def __init__(self):
        """
        Initializes the N6SectionDraftingNode.
        The LLM model is loaded dynamically in the run method or can be
        pre-loaded if configuration is static.
        """
        self.llm_model_name = settings.llm_model_name  # Utiliser le modèle configuré
        self.temperature = 0.1  # Température plus basse pour une rédaction factuelle
        self.llm: ChatOllama | None = None
        # self.output_parser = StrOutputParser() # Si la sortie est une simple chaîne

        try:
            self.llm = ChatOllama(
                model=self.llm_model_name,
                temperature=self.temperature,
                # format="json" # Seulement si la sortie est structurée et attendue en JSON par le LLM
            )
            # if using structured output:
            # self.structured_llm = self.llm.with_structured_output(schema=SectionDraftOutput)
            logger.info(
                "N6SectionDraftingNode initialized with LLM: %s", self.llm_model_name
            )
        except Exception as e:  # noqa: BLE001 - Broad exception for LLM init
            logger.error(
                "N6 Fatal: ChatOllama(%s) failed to initialize: %s",
                self.llm_model_name,
                e,
                exc_info=True,
            )
            # Le noeud ne pourra pas fonctionner sans LLM. Une erreur sera levée dans run().

    def _build_drafting_prompt_template(self) -> ChatPromptTemplate:
        """
        Builds the prompt template for the LLM to draft a section.
        """
        # fmt: off
        prompt_str = (
            "Vous êtes un rédacteur académique expert, chargé de rédiger une section spécifique "
            "d'un mémoire de fin d'études pour un étudiant d'Epitech Digital School. "
            "Le mémoire est une \"Mission Professionnelle\" pour le titre RNCP 35284 "
            "\"Expert en management des systèmes d'information\".\n\n"
            "**Persona de l'Étudiant :**\n{persona}\n\n"
            "**Section à Rédiger :**\n"
            "- Titre de la section : \"{section_title}\"\n"
            "- Objectifs de la section : {section_objectives}\n"
            "- Exigences Epitech pour cette section : {section_requirements_summary}\n"
            "- Questions clés auxquelles cette section doit répondre : {section_key_questions}\n"
            "- Notes sur le style/type de contenu (inspiré de l'exemple de mémoire) : {section_style_notes}\n\n"
            "**Contexte Pertinent Extrait du Journal de Bord de l'Étudiant (anonymisé) :**\n"
            "Utilisez ces extraits pour illustrer les propos, fournir des exemples concrets, "
            "et étayer l'analyse. Intégrez-les de manière fluide et pertinente. "
            "NE PAS simplement copier/coller les extraits. Reformulez et analysez.\n"
            "--- DÉBUT CONTEXTE JOURNAL ---\n{journal_context}\n"
            "--- FIN CONTEXTE JOURNAL ---\n\n"
            "**Instructions de Rédaction :**\n"
            "1.  Rédigez le contenu de la section en respectant un ton académique, professionnel, clair et concis.\n"
            "2.  Assurez-vous que le contenu répond directement aux objectifs, aux exigences Epitech, et aux questions clés listées ci-dessus.\n"
            "3.  Intégrez les expériences du journal de bord de manière significative. Expliquez comment ces expériences illustrent les compétences ou les concepts abordés.\n"
            "4.  **Très Important :** Le contexte du journal est déjà anonymisé. Lors de la rédaction, maintenez cette anonymisation. Ne réintroduisez AUCUN nom de personne réelle, de société spécifique (sauf GECINA si explicitement autorisé pour le contexte général de l'entreprise), ou de projet interne non public. Utilisez des termes génériques si nécessaire (ex: \"mon manager\", \"un projet spécifique\", \"l'outil X\").\n"
            "5.  Évitez les opinions personnelles \"crues\". Toute réflexion doit être présentée de manière analytique et professionnelle.\n"
            "6.  Structurez le contenu de manière logique avec des paragraphes bien définis.\n"
            "7.  Si des sources externes sont implicitement nécessaires (par exemple, pour définir un concept théorique mentionné dans le journal), signalez-le par [NÉCESSITE CITATION EXTERNE POUR : concept_X] mais ne tentez pas de générer une bibliographie ici.\n"
            "8.  La longueur doit être appropriée pour une section de mémoire, en visant la substance plutôt que le volume excessif.\n\n"
            "**Format de Sortie Attendu :**\n"
            "Rédigez UNIQUEMENT le texte du contenu de la section. Ne pas inclure le titre de la section à nouveau, ni d'en-têtes ou de formatage spécifiques (ce sera géré plus tard).\n"
            # "Si vous utilisez un schéma de sortie structuré avec SectionDraftOutput:"
            # "Répondez UNIQUEMENT avec l'objet JSON adhérant au schéma SectionDraftOutput."
        )
        # fmt: on
        return ChatPromptTemplate.from_template(prompt_str)

    def run(self, state: AgentState) -> dict[str, Any]:  # noqa: C901
        """
        Drafts a thesis section using the LLM.
        """
        logger.info("N6: Section Drafting Node starting.")
        updated_fields: dict[str, Any] = {
            "last_successful_node": "N6SectionDraftingNode_Error",  # Default in case of early exit
            "current_operation_message": "N6: Initializing section drafting.",
            "error_message": None,
        }

        if not self.llm:
            msg = "N6: LLM not initialized. Cannot proceed with drafting."
            logger.error(msg)
            updated_fields["error_message"] = msg
            return updated_fields

        current_section_id = state.current_section_id
        if not current_section_id:
            msg = "N6: current_section_id is not set in state. Cannot draft section."
            logger.error(msg)
            updated_fields["error_message"] = msg
            return updated_fields

        section_to_draft: SectionDetail | None = None
        target_section_index: int | None = None
        # Assurer que thesis_outline est une liste
        thesis_outline_list = (
            state.thesis_outline if isinstance(state.thesis_outline, list) else []
        )

        for i, section in enumerate(thesis_outline_list):
            if section.id == current_section_id:
                section_to_draft = section
                target_section_index = i
                break

        if not section_to_draft or target_section_index is None:
            msg = f"N6: Section with ID '{current_section_id}' not found in thesis_outline."
            logger.error(msg)
            updated_fields["error_message"] = msg
            updated_fields["thesis_outline"] = (
                thesis_outline_list  # Retourner la liste originale
            )
            return updated_fields

        logger.info(
            f"N6: Drafting section: '{section_to_draft.title}' (ID: {section_to_draft.id})"
        )

        if (
            not section_to_draft.anonymized_context_for_llm
            and section_to_draft.student_experience_keywords
        ):
            logger.warning(
                f"N6: Section '{section_to_draft.title}' has keywords but no "
                "retrieved context (anonymized_context_for_llm is empty). "
                "Drafting may be less informed."
            )

        # Préparer les entrées du prompt
        persona = state.user_persona
        journal_context = (
            section_to_draft.anonymized_context_for_llm
            or "[Aucun extrait de journal spécifique n'a été jugé pertinent pour cette section ou aucun mot-clé n'a été fourni.]"
        )
        style_notes = (
            section_to_draft.example_phrasing_or_content_type
            or "Style académique standard."
        )

        prompt_template = self._build_drafting_prompt_template()
        prompt_values = {
            "persona": persona,
            "section_title": section_to_draft.title,
            "section_objectives": section_to_draft.description_objectives,
            "section_requirements_summary": section_to_draft.original_requirements_summary,
            "section_key_questions": "\n- ".join(
                section_to_draft.key_questions_to_answer
            )
            if section_to_draft.key_questions_to_answer
            else "N/A",
            "section_style_notes": style_notes,
            "journal_context": journal_context,
        }

        try:
            formatted_prompt = prompt_template.format_prompt(**prompt_values)
            logger.debug("N6: Prompt for LLM:\n%s", formatted_prompt.to_string())

            llm_response = self.llm.invoke(formatted_prompt)
            draft_text = (
                llm_response.content
                if hasattr(llm_response, "content")
                else str(llm_response)
            )

            # Mettre à jour la section dans une copie de thesis_outline
            # Pydantic V1 models are mutable, so modifying section_to_draft *could* modify it in state.thesis_outline
            # if it's the same object. Creating a new list of sections is safer.
            new_thesis_outline = [s.copy(deep=True) for s in thesis_outline_list]

            section_in_new_outline = new_thesis_outline[target_section_index]
            section_in_new_outline.draft_v1 = draft_text.strip()
            section_in_new_outline.status = SectionStatus.DRAFT_GENERATED

            updated_fields["thesis_outline"] = new_thesis_outline
            updated_fields["current_operation_message"] = (
                f"N6: Draft generated for section '{section_to_draft.title}'."
            )
            updated_fields["last_successful_node"] = "N6SectionDraftingNode"
            logger.info(
                f"N6: Draft generated for section '{section_to_draft.title}'. Length: {len(draft_text)} chars."
            )

        except Exception as e:  # noqa: BLE001 - Broad exception for LLM call or processing
            error_msg = f"N6: Error during LLM call or processing for section '{section_to_draft.title}': {e}"
            logger.error(error_msg, exc_info=True)
            updated_fields["error_message"] = error_msg

            new_thesis_outline_on_error = [
                s.copy(deep=True) for s in thesis_outline_list
            ]
            section_in_new_outline_error = new_thesis_outline_on_error[
                target_section_index
            ]
            section_in_new_outline_error.status = SectionStatus.ERROR
            section_in_new_outline_error.error_details_n6_drafting = str(e)
            updated_fields["thesis_outline"] = new_thesis_outline_on_error

        return updated_fields
