# src/nodes/n6_section_drafting.py
import logging
import traceback
from typing import Any

from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from src.config import settings
from src.state import AgentState, SectionDetail, SectionStatus

logger = logging.getLogger(__name__)


class N6SectionDraftingNode:
    """
    Node responsible for drafting the content of a specific thesis section.

    This node uses the LLM to generate a draft based on the section plan,
    retrieved context from the journal, and overall academic guidelines.
    It can also revise a draft based on a critique from N7.
    """

    def __init__(self):
        """
        Initializes the N6SectionDraftingNode.
        """
        self.llm_model_name = settings.llm_model_name
        self.temperature = 0.1
        self.llm: ChatOllama | None = None

        try:
            self.llm = ChatOllama(
                model=self.llm_model_name,
                temperature=self.temperature,
            )
            logger.info(
                "N6SectionDraftingNode initialized with LLM: %s", self.llm_model_name
            )
        except Exception as e:  # noqa: BLE001
            logger.error(
                "N6 Fatal: ChatOllama(%s) failed to initialize: %s",
                self.llm_model_name,
                e,
                exc_info=True,
            )

    def _build_initial_draft_prompt_template(self) -> ChatPromptTemplate:  # Renamed
        """
        Builds the prompt template for the LLM to draft an initial section.
        """
        # fmt: off
        prompt_str = (
            "Vous êtes un rédacteur académique chevronné et un analyste de recherche, expert dans la transformation "
            "d'expériences professionnelles concrètes, de plans de section détaillés, et d'extraits de journal de bord en contenu de mémoire de fin d'études "
            "rigoureux et bien argumenté pour un étudiant d'Epitech Digital School. Le mémoire est une \"Mission Professionnelle\" pour le titre "
            "RNCP 35284 \"Expert en management des systèmes d'information\". Votre travail doit être d'une qualité irréprochable.\n\n"

            "**Persona de l'Étudiant (pour contextualiser les expériences du journal) :**\n{persona}\n\n"

            "**Détails Impératifs de la Section Spécifique à Rédiger (issus du plan N3) :**\n"
            "- Titre Exact : \"{section_title}\"\n"
            "- Niveau Hiérarchique (informatif) : {section_level}\n"
            "- Objectifs Clés de cette Section : {section_objectives}\n"
            "- Exigences Spécifiques Epitech pour cette Section (Résumé) : {section_requirements_summary}\n"
            "- Questions Fondamentales auxquelles cette Section DOIT Répondre : {section_key_questions}\n"
            "- Notes sur le Style/Type de Contenu Attendu (inspiré de l'exemple Digi5) : {section_style_notes}\n\n"

            "**Extraits Pertinents du Journal de Bord de l'Étudiant (anonymisés, fournis par N5 après recherche RAG basée sur les keywords de N3) :**\n"
            "Ces extraits sont la **matière première essentielle** de votre rédaction. Ils représentent les expériences vécues, les tâches accomplies, les outils utilisés, les défis rencontrés, et les réflexions personnelles de l'étudiant qui sont jugés pertinents pour CETTE section. Votre mission n'est PAS de les résumer ou de les paraphraser superficiellement. Vous devez les **analyser en profondeur, les interpréter, les connecter aux objectifs de la section, aux exigences Epitech, et aux compétences RNCP (si pertinent pour cette section), et les intégrer de manière fluide, analytique et judicieuse dans votre argumentation.**\n"
            "--- DÉBUT CONTEXTE JOURNAL ---\n{journal_context}\n"
            "--- FIN CONTEXTE JOURNAL ---\n\n"

            "**Instructions Spécifiques pour la Rédaction de cette Section :**\n"
            "1.  **Ton et Style Académique Exigeant :** Adoptez un ton formel, académique, professionnel, objectif, analytique et réflexif. La rédaction doit être d'une clarté exemplaire, concise, précise et rigoureuse. Évitez toute familiarité ou opinion non fondée.\n"
            "2.  **Réponse Exhaustive aux Objectifs et Questions :** Le contenu rédigé doit répondre directement, explicitement et de manière exhaustive à TOUS les objectifs de la section, à TOUTES les exigences Epitech spécifiées, et à TOUTES les questions clés listées pour cette section.\n"
            "3.  **Intégration Profonde et Analytique du Journal (CRUCIAL) :**\n"
            "    - Chaque extrait du journal pertinent doit être utilisé comme une **preuve tangible**, une **illustration concrète**, ou un **point de départ pour une analyse** en lien avec les points que vous développez.\n"
            "    - **Explicitez le lien :** Montrez clairement au lecteur comment ces expériences (projets, tâches, outils, réflexions du journal) sont pertinentes pour la section, comment elles démontrent les compétences, valident un concept, ou illustrent un défi/solution.\n"
            "    - Si la section concerne l'analyse de compétences RNCP, chaque exemple tiré du journal doit être explicitement relié à la (ou aux) compétence(s) RNCP spécifique(s) visée(s) par cette section (mentionnées dans `section_requirements_summary`).\n"
            "    - **Ne vous contentez pas d'insérer des citations du journal.** Intégrez l'information, analysez-la, et commentez sa signification dans le contexte de votre argumentation.\n"
            "4.  **Maintien Strict de l'Anonymisation :** Le contexte du journal fourni est déjà anonymisé. Vous devez impérativement maintenir cette anonymisation. N'introduisez AUCUN nom de personne réelle (utilisez les placeholders fournis ou des rôles génériques comme '[Le Directeur de Projet]', '[Un Membre de l'Équipe Technique]'), de société spécifique (sauf 'GECINA' si explicitement autorisé pour le contexte général de l'entreprise d'accueil), ou de nom de projet interne non public et non-anonymisé. En cas de doute, utilisez un terme générique et signalez-le (voir point 7).\n"
            "5.  **Analyse, Réflexion et Justification :** Toute affirmation doit être étayée. Toute réflexion personnelle de l'étudiant (tirée du journal ou implicite) doit être présentée et analysée de manière professionnelle et distanciée. Justifiez vos analyses et conclusions.\n"
            "6.  **Structure, Cohérence et Fluidité :** Organisez le contenu en paragraphes logiques et bien articulés. Utilisez des phrases de transition claires pour assurer une lecture fluide et une argumentation progressive et facile à suivre. Chaque paragraphe doit développer une idée principale.\n"
            "7.  **Identification des Besoins de Citations Externes :** Si, pour étayer une affirmation, définir un concept théorique clé, ou contextualiser une pratique mentionnée dans le journal, une référence à une source externe (article académique, ouvrage de référence, standard industriel, etc.) est manifestement nécessaire et absente du contexte fourni, **indiquez-le très clairement et précisément dans le texte** en utilisant le format suivant : `[NÉCESSITE CITATION EXTERNE POUR : décrire précisément le concept, l'affirmation ou l'information à sourcer. Exemple : 'définition du framework Agile Scrum' ou 'statistiques sur l'adoption de l'IA dans le secteur foncier']`. Ne pas inventer de sources.\n"
            "8.  **Longueur, Profondeur et Substance :** Rédigez un contenu substantiel, approfondi et approprié pour une section de mémoire de niveau Master. La qualité et la profondeur de l'analyse priment sur la quantité pure. Une sous-section typique peut faire plusieurs paragraphes consistants. Un chapitre peut s'étendre sur plusieurs pages. Assurez-vous de couvrir tous les aspects demandés pour la section.\n\n"

            "**Format de Sortie Attendu (Strict) :**\n"
            "Rédigez **UNIQUEMENT et DIRECTEMENT le texte du contenu de la section elle-même.**\n"
            "Ne PAS inclure le titre de la section (il est déjà connu).\n"
            "Ne PAS inclure d'en-têtes, de numérotation de section, ou de formatage spécifique (gras, italique, etc. – le formatage sera géré ultérieurement).\n"
            "Commencez directement par le premier mot du premier paragraphe du contenu de la section. Terminez par le dernier mot du dernier paragraphe."
        )
        # fmt: on
        return ChatPromptTemplate.from_template(prompt_str)

    def _build_revision_prompt_template(self) -> ChatPromptTemplate:
        """
        Builds the prompt template for the LLM to revise a section based on critique.
        """
        # fmt: off
        prompt_str = (
            "Vous êtes un rédacteur académique expert et un spécialiste de l'amélioration de texte, chargé de **réviser et d'améliorer** un brouillon de section de mémoire professionnel Epitech. Vous devez vous baser sur une critique détaillée qui a été générée précédemment.\n\n"
            "**Persona de l'Étudiant (Contexte pour les expériences du journal) :**\n{persona}\n\n"
            "**Détails de la Section à Réviser :**\n"
            "- Titre Exact : \"{section_title}\"\n"
            "- Niveau Hiérarchique : {section_level}\n"
            "- Objectifs Clés : {section_objectives}\n"
            "- Exigences Epitech : {section_requirements_summary}\n"
            "- Questions Fondamentales : {section_key_questions}\n"
            "- Notes sur Style/Contenu : {section_style_notes}\n\n"
            "**Brouillon Précédent à Réviser (`draft_v{previous_draft_version}`) :**\n"
            "--- DÉBUT BROUILLON PRÉCÉDENT ---\n{previous_draft_content}\n"
            "--- FIN BROUILLON PRÉCÉDENT ---\n\n"
            "**Critique Structurée du Brouillon Précédent (générée par N7_SelfCritiqueNode) :**\n"
            "Cette critique détaille les points faibles, les manques, et les éléments superflus du brouillon précédent. Vous devez **impérativement adresser chaque point de cette critique** dans votre version révisée.\n"
            "--- DÉBUT CRITIQUE ---\n{critique_json_str}\n" # critique_json_str sera le .json() de CritiqueOutput
            "--- FIN CRITIQUE ---\n\n"
            "**Contexte du Journal d'Apprentissage (si des informations supplémentaires ont été suggérées par la critique et récupérées) :**\n"
            "(Ce contexte peut être le même que pour le draft initial, ou enrichi si la critique a mené à de nouvelles recherches RAG)\n"
            "--- DÉBUT CONTEXTE JOURNAL ACTUALISÉ ---\n{journal_context}\n"
            "--- FIN CONTEXTE JOURNAL ACTUALISÉ ---\n\n"
            "**Votre Tâche de Révision :**\n"
            "1.  **Analysez attentivement la critique** (`critique_json_str`) et identifiez tous les points d'amélioration demandés.\n"
            "2.  **Réécrivez intégralement** la section pour produire un nouveau brouillon (`refined_draft` ou `draft_v(X+1)`) qui adresse **tous les points soulevés dans la critique**.\n"
            "3.  Conservez les forces du brouillon précédent qui n'ont pas été critiquées.\n"
            "4.  Assurez-vous que la version révisée respecte **toutes les instructions de rédaction initiales** (ton, style, intégration du journal, anonymisation, réponse aux objectifs, signalement de citations, etc.) en plus des corrections demandées.\n"
            "5.  Si la critique suggère d'intégrer de nouvelles informations du journal (suite à `suggested_search_queries`), utilisez le `journal_context` (potentiellement actualisé) pour cela.\n"
            "6.  Soyez particulièrement attentif à corriger les `identified_flaws`, à combler les `missing_information`, et à retirer le `superfluous_content`.\n\n"
            "**Format de Sortie Attendu (Strict) :**\n"
            "Rédigez **UNIQUEMENT et DIRECTEMENT le texte du contenu de la section révisée.**\n"
            "Ne PAS inclure le titre de la section.\n"
            "Ne PAS inclure d'en-têtes ou de formatage spécifique.\n"
            "Commencez directement par le premier mot du premier paragraphe. Terminez par le dernier mot du dernier paragraphe."
        )
        # fmt: on
        return ChatPromptTemplate.from_template(prompt_str)

    def run(self, state: AgentState) -> dict[str, Any]:  # noqa: C901
        """
        Drafts or revises a thesis section using the LLM.
        """
        logger.info("N6: Section Drafting/Revising Node starting.")
        updated_fields: dict[str, Any] = {
            "last_successful_node": "N6SectionDraftingNode_Error",
            "current_operation_message": "N6: Initializing section drafting/revision.",
            "error_message": None,
            "error_details_n6_drafting": None,
        }

        if not self.llm:
            msg = "N6: LLM not initialized. Cannot proceed."
            logger.error(msg)
            updated_fields["error_message"] = msg
            return updated_fields

        current_section_id = state.current_section_id
        if not current_section_id:
            msg = "N6: current_section_id is not set. Cannot proceed."
            logger.error(msg)
            updated_fields["error_message"] = msg
            return updated_fields

        section_to_process: SectionDetail | None = None
        target_section_index: int | None = None
        thesis_outline_list = (
            state.thesis_outline if isinstance(state.thesis_outline, list) else []
        )

        for i, section in enumerate(thesis_outline_list):
            if section.id == current_section_id:
                section_to_process = section  # Référence à l'objet dans la liste
                target_section_index = i
                break

        if not section_to_process or target_section_index is None:
            msg = (
                f"N6: Section with ID '{current_section_id}' not found "
                "in thesis_outline."
            )
            logger.error(msg)
            updated_fields["error_message"] = msg
            # Pas besoin de retourner thesis_outline ici car on n'a pas pu le modifier
            return updated_fields

        # Travailler sur une copie pour éviter de modifier l'état directement dans la boucle
        # avant que toutes les opérations LLM ne soient terminées.
        section_data_for_prompt = section_to_process.copy(deep=True)

        persona = state.user_persona
        journal_context = (
            section_data_for_prompt.anonymized_context_for_llm
            or "[Aucun extrait de journal spécifique n'a été jugé pertinent pour cette section ou aucun mot-clé n'a été fourni.]"
        )
        style_notes = (
            section_data_for_prompt.example_phrasing_or_content_type
            or "Style académique standard, analytique et réflexif."
        )
        key_questions_str = (
            "\n- ".join(section_data_for_prompt.key_questions_to_answer)
            if section_data_for_prompt.key_questions_to_answer
            else "N/A"
        )

        prompt_values = {
            "persona": persona,
            "section_title": section_data_for_prompt.title,
            "section_level": section_data_for_prompt.level,
            "section_objectives": section_data_for_prompt.description_objectives,
            "section_requirements_summary": section_data_for_prompt.original_requirements_summary,
            "section_key_questions": key_questions_str,
            "section_style_notes": style_notes,
            "journal_context": journal_context,
        }

        # Déterminer si c'est un draft initial ou une révision
        # La critique est dans section_to_process.critique_v1 (l'original, pas la copie)
        # ou on pourrait passer la critique via un champ dédié dans AgentState si N7 est séparé.
        # Pour l'instant, on se base sur la présence d'une critique dans la section
        # ou sur un flag spécifique si on utilise un sous-graphe Reflexion.

        # Pour cet exemple, on va supposer que N7 met à jour section_to_process.critique_v1
        # et que N6 est rappelé. N6 vérifie alors ce champ.
        # Le `current_draft_for_critique` pourrait être le `draft_v1` ou un `refined_draft` antérieur.

        is_revision_mode = (
            section_to_process.critique_v1 is not None
            and section_to_process.status == SectionStatus.SELF_CRITIQUE_COMPLETED
        )
        # Ou un autre statut indiquant une demande de révision après critique

        current_draft_content_for_revision = (
            section_to_process.current_draft_for_critique
        )

        if (
            is_revision_mode
            and current_draft_content_for_revision
            and section_to_process.critique_v1
        ):
            logger.info(
                "N6: Revising section: '%s' based on critique.",
                section_data_for_prompt.title,
            )
            prompt_template = self._build_revision_prompt_template()
            prompt_values["previous_draft_version"] = (
                section_to_process.reflection_attempts
            )  # ou une autre logique de versionnement
            prompt_values["previous_draft_content"] = current_draft_content_for_revision
            prompt_values["critique_json_str"] = (
                section_to_process.critique_v1.json()
            )  # Pydantic V1
        else:
            logger.info(
                "N6: Drafting initial version for section: '%s' (ID: %s, Level: %s)",
                section_data_for_prompt.title,
                section_data_for_prompt.id,
                section_data_for_prompt.level,
            )
            prompt_template = self._build_initial_draft_prompt_template()
            # Réinitialiser les champs liés à la critique/révision pour un premier draft
            section_data_for_prompt.critique_v1 = None
            section_data_for_prompt.refined_draft = None
            section_data_for_prompt.reflection_history = []
            section_data_for_prompt.reflection_attempts = 0

        if (
            not section_data_for_prompt.anonymized_context_for_llm
            and section_data_for_prompt.student_experience_keywords
            and not is_revision_mode  # Ne pas re-logger si c'est une révision
        ):
            logger.warning(
                f"N6: Section '{section_data_for_prompt.title}' has keywords but no "
                "retrieved context. Drafting may be less informed."
            )

        try:
            formatted_prompt = prompt_template.format_prompt(**prompt_values)
            logger.debug(
                "N6: Prompt for LLM (%s):\n%s",
                "Revision" if is_revision_mode else "Initial Draft",
                formatted_prompt.to_string()[:1000] + "...",
            )

            llm_response = self.llm.invoke(formatted_prompt)
            generated_text = (
                llm_response.content
                if hasattr(llm_response, "content")
                else str(llm_response)
            ).strip()

            # Mettre à jour la copie de la section avant de l'assigner à l'outline
            # et ensuite mettre à jour l'état.
            # Utiliser une nouvelle liste pour thesis_outline pour éviter les mutations inattendues
            new_thesis_outline = [s.copy(deep=True) for s in thesis_outline_list]
            section_to_update_in_new_outline = new_thesis_outline[target_section_index]

            if is_revision_mode:
                section_to_update_in_new_outline.refined_draft = generated_text
                # Le statut sera géré par la boucle de réflexion (N7 ou le routeur)
                # Typiquement, après une révision, on repasse par N7.
                # Ici, on marque comme DRAFT_GENERATED en attendant la prochaine critique.
                section_to_update_in_new_outline.status = SectionStatus.DRAFT_GENERATED
                updated_fields["current_operation_message"] = (
                    f"N6: Revision generated for section '{section_data_for_prompt.title}'."
                )
            else:
                section_to_update_in_new_outline.draft_v1 = generated_text
                section_to_update_in_new_outline.status = SectionStatus.DRAFT_GENERATED
                updated_fields["current_operation_message"] = (
                    f"N6: Initial draft generated for section '{section_data_for_prompt.title}'."
                )

            # Mettre à jour le draft courant pour la prochaine critique potentielle
            section_to_update_in_new_outline.current_draft_for_critique = generated_text

            updated_fields["thesis_outline"] = new_thesis_outline
            updated_fields["last_successful_node"] = "N6SectionDraftingNode"
            logger.info(
                "N6: %s for section '%s'. Length: %d chars.",
                "Revision generated" if is_revision_mode else "Initial draft generated",
                section_data_for_prompt.title,
                len(generated_text),
            )

        except Exception as e:  # noqa: BLE001
            error_msg = (
                f"N6: Error during LLM call or processing for section "
                f"'{section_data_for_prompt.title}': {e}"
            )
            logger.error(error_msg, exc_info=True)
            updated_fields["error_message"] = error_msg
            error_details_full = f"{error_msg}\n{traceback.format_exc()}"
            updated_fields["error_details_n6_drafting"] = error_details_full

            # Mettre à jour l'état d'erreur sur la section dans une nouvelle copie de l'outline
            new_thesis_outline_on_error = [
                s.copy(deep=True) for s in thesis_outline_list
            ]
            section_in_new_outline_error = new_thesis_outline_on_error[
                target_section_index
            ]
            section_in_new_outline_error.status = SectionStatus.ERROR
            section_in_new_outline_error.error_details_n6_drafting = error_details_full
            updated_fields["thesis_outline"] = new_thesis_outline_on_error
            # Ne pas écraser last_successful_node s'il y a une erreur ici

        return updated_fields
