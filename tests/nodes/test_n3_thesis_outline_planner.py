# tests/nodes/test_n3_thesis_outline_planner.py
import logging
import unittest
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage

from src.nodes.n3_thesis_outline_planner import (
    N3ThesisOutlinePlannerNode,
    PlannedSectionDetailForLLM,
    PlannedThesisOutlineForLLM,
)
from src.state import AgentState, SectionDetail, SectionStatus

logging.basicConfig(level=logging.INFO)
logging.getLogger("src.nodes.n3_thesis_outline_planner").setLevel(logging.DEBUG)


EXAMPLE_THESIS_DIGI5_ABRIDGED_FOR_TEST = """
## AVANT-PROPOS
Ceci est un avant-propos.
## SOMMAIRE
Ceci est un sommaire.
## INTRODUCTION GÉNÉRALE
### 1. Contexte de la mission
Le contexte était stimulant.
### 2. Problématique et Objectifs du mémoire
La problématique était complexe. Les objectifs étaient clairs.
## CHAPITRE 1 : PRÉSENTATION DE GECINA ET DE LA MISSION
### 1.1. L'entreprise GECINA
Gecina est une grande foncière. Description de son activité, son positionnement sur le marché immobilier, et sa stratégie de développement durable CAN0P-2030. Mention de ses services comme Yourplace et FEAT.
### 1.2. Ma mission et mon rôle
Ma mission était de développer l'IA chez Gecina. J'étais AIPO, rattaché à Jérôme Carecchio. J'ai travaillé sur le prompt engineering pour les RH, et sur une solution Power Automate pour la DG.
## CHAPITRE 3 : ANALYSE DES COMPÉTENCES (RNCP35284)
### 3.1. Bloc 1: Manager un projet Informatique et Systèmes d'Information
#### 3.1.1. Activité : Analyser les besoins des utilisateurs
Contenu: Projet X - Analyse des besoins clients pour un chatbot. Projet Alpha, analyse des besoins pour la Direction Financière, utilisation de Power BI pour des dashboards. J'ai identifié les KPIs.
#### 3.1.2. Activité : Concevoir la solution technique
Contenu: Projet Y - Conception de l'architecture d'un outil de prédiction. Solution Power Automate pour la DG: architecture, OCR, AI Builder, fonction Azure.
...
"""

# REALISTIC_MOCK_PLANNED_SECTIONS (Version from around 24 mai 23:36 UTC)
REALISTIC_MOCK_PLANNED_SECTIONS: list[PlannedSectionDetailForLLM] = [
    PlannedSectionDetailForLLM(
        id="0.1.",
        title="Avant-propos / Remerciements",
        level=1,
        description_objectives=(
            "Section pour exprimer la gratitude envers les personnes et entités "
            "ayant soutenu la réalisation du mémoire et de l'alternance, et pour "
            "introduire brièvement le contexte personnel de cette démarche."
        ),
        original_requirements_summary=(
            "Optionnelle selon les directives Epitech, mais présente dans "
            "l'exemple Digi5. Permet de contextualiser le travail et de "
            "remercier les contributeurs."
        ),
        student_experience_keywords=[
            "remerciements tuteur entreprise M. Carecchio",
            "soutien pédagogique équipe Epitech",
            "contribution collègues projet Power Automate",
            "opportunité d'alternance chez GECINA",
            "expérience chef de projet IA",
            "apprentissage continu et défis",
        ],
        example_phrasing_or_content_type=(
            "Ton personnel et professionnel, sincère. Courte introduction et liste "
            "de remerciements nominatifs ou par groupe."
        ),
        key_questions_to_answer=[
            "Qui a joué un rôle clé dans mon parcours et la réalisation de ce mémoire ?",
            "Comment exprimer ma gratitude de manière formelle et sincère ?",
        ],
    ),
    PlannedSectionDetailForLLM(
        id="0.2.",
        title="Sommaire Détaillé",
        level=1,
        description_objectives=(
            "Fournir une table des matières claire et hiérarchisée du mémoire, "
            "permettant une navigation aisée pour le lecteur."
        ),
        original_requirements_summary=(
            "Section standard et indispensable pour tout document académique long. "
            "Doit refléter fidèlement la structure finale du mémoire."
        ),
        student_experience_keywords=[
            "élaboration plan mémoire",
            "structuration des idées",
            "organisation des chapitres",
            "logique de progression",
            "table des matières provisoire",
            "hiérarchie des sections",
        ],
        example_phrasing_or_content_type=(
            "Liste hiérarchique des titres de chapitres, sections et sous-sections "
            "avec numérotation et pagination (la pagination sera ajoutée ultérieurement)."
        ),
        key_questions_to_answer=[
            "Quelles sont toutes les parties principales et secondaires de mon mémoire ?",
            (
                "La numérotation et la hiérarchie des sections sont-elles logiques et "
                "conformes aux standards ?"
            ),
        ],
    ),
    PlannedSectionDetailForLLM(
        id="1.",
        title="Introduction Générale",
        level=1,
        description_objectives=(
            "Présenter le contexte général de la mission en alternance au sein de "
            "GECINA, introduire la problématique centrale du mémoire liée à "
            "l'intégration de l'IA dans le secteur immobilier, définir les objectifs "
            "visés (notamment en lien avec le titre RNCP 35284), et annoncer le plan "
            "du document."
        ),
        original_requirements_summary=(
            "Les directives Epitech insistent sur la clarté de la problématique et "
            "des objectifs. L'exemple Digi5 montre une introduction structurée "
            "(contexte, problématique, objectifs)."
        ),
        student_experience_keywords=[
            "immersion contexte GECINA",
            "identification problématique IA immobilier",
            "définition objectifs mémoire RNCP 35284",
            "structuration du plan de mémoire",
            "enjeux transformation digitale secteur foncier",
            "mon rôle d'AIPO (Chef de Projet IA)",
            "innovation par l'IA chez GECINA",
        ],
        example_phrasing_or_content_type=(
            "Style formel, académique, clair et concis. Doit capter l'intérêt du "
            "lecteur et justifier l'importance du sujet traité."
        ),
        key_questions_to_answer=[
            "Quel était le contexte de ma mission chez GECINA ?",
            "Quelle problématique centrale ce mémoire aborde-t-il concernant l'IA ?",
            (
                "Quels sont les objectifs de ce mémoire et quelle structure vais-je suivre "
                "pour y répondre ?"
            ),
        ],
    ),
    PlannedSectionDetailForLLM(
        id="1.1.",
        title="Description de l'entreprise d'accueil : GECINA",
        level=2,
        description_objectives=(
            "Présenter en détail GECINA : son statut de SIIC, son secteur d'activité "
            "(principalement l'immobilier de bureaux), son positionnement sur le marché "
            "(focus Paris et grands pôles tertiaires), sa stratégie de développement "
            "durable (ex: CAN0P-2030) et ses services innovants (ex: Yourplace, FEAT)."
        ),
        original_requirements_summary=(
            "Déduit de Digi5 et des directives Epitech qui demandent de contextualiser "
            "la mission. Cette section est importante pour comprendre le cadre de "
            "l'alternance."
        ),
        student_experience_keywords=[
            "présentation GECINA",
            "statut SIIC",
            "marché immobilier de bureaux Paris",
            "stratégie RSE CAN0P-2030",
            "services innovants Yourplace FEAT",
            "organisation interne GECINA",
            "culture d'entreprise GECINA",
            "actifs immobiliers GECINA",
        ],
        example_phrasing_or_content_type=(
            "Contenu factuel, analytique, précis. Utiliser des informations publiques "
            "si possible (rapports annuels, site web). S'inspirer du style descriptif "
            "de Digi5 pour cette section."
        ),
        key_questions_to_answer=[
            (
                "Qu'est-ce que GECINA, son activité principale et son statut juridique et "
                "financier ?"
            ),
            (
                "Comment GECINA se positionne-t-elle sur son marché et quelle est sa "
                "stratégie en matière de RSE et d'innovation technologique ?"
            ),
        ],
    ),
    PlannedSectionDetailForLLM(
        id="1.2.",
        title=(
            "Ma mission en tant que Chef de Projet IA (AIPO) et objectifs du " "mémoire"
        ),
        level=2,
        description_objectives=(
            "Décrire en détail ma mission en tant qu'AIPO (Artificial Intelligence "
            "Project Officer) chez Gecina, les projets IA principaux sur lesquels "
            "j'ai travaillé (ex: 'Projet Prompt Engineering RH', 'Projet Automatisation "
            "DG via Power Automate'), mon positionnement hiérarchique (rattachement à "
            "Jérôme Carecchio). Clarifier les objectifs spécifiques de ce mémoire en "
            "lien avec le titre RNCP et l'auto-évaluation de mes compétences."
        ),
        original_requirements_summary=(
            "Les directives Epitech exigent de lier le mémoire à la mission "
            "professionnelle et aux compétences du référentiel RNCP. L'exemple Digi5 "
            "détaille la mission et son contexte."
        ),
        student_experience_keywords=[
            "rôle AIPO GECINA",
            "projets IA GECINA",
            "prompt engineering RH",
            "solution Power Automate DG",
            "utilisation OCR, AI Builder, fonction Azure",
            "rattachement Jérôme Carecchio",
            "objectifs RNCP 35284",
            "validation des compétences",
            "auto-évaluation AIPO",
            "contribution à la stratégie IA de GECINA",
        ],
        example_phrasing_or_content_type=(
            "Style descriptif et analytique. Mettre en avant les responsabilités "
            "assumées, les technologies utilisées et les enjeux des projets menés. "
            "Énoncer clairement les buts du mémoire et sa contribution à la validation "
            "du titre RNCP."
        ),
        key_questions_to_answer=[
            (
                "Quelle était ma mission principale et mes responsabilités concrètes en "
                "tant qu'AIPO chez GECINA ?"
            ),
            (
                "Quels sont les objectifs de ce mémoire, notamment par rapport au RNCP "
                "35284 et à mon développement de compétences en IA ?"
            ),
        ],
    ),
    PlannedSectionDetailForLLM(
        id="3.",
        title="Analyse des Compétences (RNCP 35284)",
        level=1,
        description_objectives=(
            "Analyser de manière approfondie et structurée les compétences du "
            "référentiel RNCP 35284 (Expert en management des systèmes d'information) "
            "que j'ai développées ou mobilisées durant mon alternance, en fournissant "
            "des exemples concrets de projets ('Projet Prompt Engineering RH', "
            "'Projet Automatisation DG') et de tâches réalisées chez GECINA pour chaque "
            "compétence ou activité clé."
        ),
        original_requirements_summary=(
            "Section cruciale selon les directives Epitech. Doit couvrir les blocs de "
            "compétences RNCP (Analyser les besoins, Concevoir la solution, Gérer les "
            "projets SI, Maintenir le SI, Assister et former les utilisateurs). "
            "L'exemple Digi5 adopte une structure par bloc."
        ),
        student_experience_keywords=[
            "compétences RNCP 35284",
            "analyse de besoins utilisateurs IA",
            "conception de la solution Power Automate",
            "gestion de projet d'automatisation de processus",
            "maintenance évolutive du chatbot RH",
            "création de support utilisateurs pour outil Power BI",
            "veille technologique sur l'IA appliquée à l'immobilier",
            "application des compétences RNCP en entreprise",
            "preuves de compétence projets",
            "difficultés rencontrées et solutions",
        ],
        example_phrasing_or_content_type=(
            "Structurer par bloc de compétences RNCP, puis par activité clé au sein de "
            "chaque bloc, comme dans l'exemple Digi5. Pour chaque activité, utiliser "
            "une méthode descriptive telle que STAR (Situation, Tâche, Action, "
            "Résultat) ou une narration de projet pour illustrer avec des exemples "
            "concrets et précis."
        ),
        key_questions_to_answer=[
            (
                "Quelles compétences spécifiques du référentiel RNCP 35284 ai-je mises "
                "en œuvre et développées chez GECINA ?"
            ),
            (
                "Comment mes expériences professionnelles (projets, tâches spécifiques) "
                "illustrent-elles concrètement l'acquisition et l'application de ces "
                "compétences ?"
            ),
            (
                "Quels ont été les résultats et les apprentissages pour chaque compétence "
                "mobilisée ?"
            ),
        ],
    ),
    PlannedSectionDetailForLLM(
        id="3.1.",
        title="Bloc 1: Manager un projet Informatique et Systèmes d'Information",
        level=2,
        description_objectives=(
            "Détailler les activités et compétences relevant du Bloc 1 du RNCP35284, "
            "en particulier 'Analyser les besoins des utilisateurs et des "
            "organisations en matière de SI' et 'Concevoir la solution technique et "
            "fonctionnelle d'un projet SI'. Illustrer avec les projets d'IA "
            "spécifiques menés chez GECINA, comme le 'Projet Automatisation DG'."
        ),
        original_requirements_summary=(
            "RNCP35284 - Bloc 1. L'exemple Digi5 structure ce bloc par activités "
            "spécifiques. Mettre l'accent sur la dimension managériale du projet SI."
        ),
        student_experience_keywords=[
            "management de projet SI",
            "analyse des besoins utilisateurs projet Automatisation DG",
            "rédaction de spécifications fonctionnelles projet RH",
            "conception de l'architecture technique solution Power Automate",
            "étude de faisabilité projet IA",
            "sélection des technologies (Power Platform, Azure AI Services)",
            "application de méthodologies de gestion de projet (Agile light)",
            "planification des étapes projet IA",
            "gestion des risques et des dépendances projet SI",
            "communication avec les parties prenantes",
        ],
        example_phrasing_or_content_type=(
            "Présenter chaque activité clé du bloc. Pour chaque activité, décrire les "
            "projets spécifiques où elle a été mise en œuvre, en détaillant les "
            "actions managériales, les outils de gestion de projet utilisés, et les "
            "résultats obtenus en termes de pilotage."
        ),
        key_questions_to_answer=[
            (
                "Comment ai-je analysé les besoins des utilisateurs et de GECINA pour les "
                "projets SI que j'ai managés ou auxquels j'ai contribué activement ?"
            ),
            (
                "Quelles solutions techniques et fonctionnelles ai-je conçues et comment "
                "ai-je assuré leur adéquation avec les besoins et les contraintes du projet ?"
            ),
        ],
    ),
    PlannedSectionDetailForLLM(
        id="3.1.1.",
        title=(
            "Activité : Analyser les besoins des utilisateurs et des organisations "
            "en matière de Systèmes d'Information"
        ),
        level=3,
        description_objectives=(
            "Illustrer par des exemples concrets ('Projet RH', 'Projet Automatisation "
            "DG') comment j'ai recueilli, évalué, formalisé et modélisé les besoins "
            "des utilisateurs (ex: Direction Financière pour Power BI, département RH "
            "pour le prompt engineering) et de l'organisation GECINA pour des projets "
            "de Systèmes d'Information basés sur l'IA."
        ),
        original_requirements_summary=(
            "Déduit de la structure de Digi5 (Projet X, Projet Alpha) et du contenu "
            "du Bloc 1 du RNCP. Met l'accent sur l'analyse des besoins spécifiques "
            "des utilisateurs et de l'organisation."
        ),
        student_experience_keywords=[
            "analyse besoins projet Alpha CRM",
            "recueil besoins Direction Financière Power BI",
            "identification KPIs reporting financier",
            "modélisation processus métier service RH",
            "animation ateliers utilisateurs chatbot",
            "rédaction spécifications fonctionnelles détaillées (SFD)",
            "analyse des pain points utilisateurs actuels",
            "priorisation des fonctionnalités backlog",
            "étude d'opportunité IA",
            "cartographie des besoins",
        ],
        example_phrasing_or_content_type=(
            "Décrire un ou plusieurs projets spécifiques (ex: Projet Alpha, projet "
            "Power Automate DG). Expliquer les méthodes et outils utilisés pour "
            "l'analyse des besoins (ateliers de co-conception, entretiens individuels, "
            "analyse documentaire, mind mapping, BPMN pour modélisation de processus) "
            "et les livrables produits (ex: comptes-rendus d'ateliers, spécifications "
            "fonctionnelles, user stories, diagrammes de processus)."
        ),
        key_questions_to_answer=[
            (
                "Quels étaient les besoins utilisateurs et organisationnels identifiés pour "
                "le projet X (ex: Alpha) ou le projet Y (ex: Power Automate DG) ?"
            ),
            ("Comment ai-je procédé pour recueillir et analyser ces besoins ?"),
            (
                "Quels ont été les livrables de cette phase d'analyse et comment ont-ils "
                "orienté la conception de la solution SI ?"
            ),
        ],
    ),
    PlannedSectionDetailForLLM(
        id="5.",
        title="Conclusion Générale",
        level=1,
        description_objectives=(
            "Synthétiser les principaux apports du mémoire, répondre de manière concise "
            "à la problématique posée en introduction, discuter des limites du travail "
            "réalisé et des projets menés, et proposer des perspectives d'évolution "
            "futures (personnelles, pour l'entreprise, et pour le domaine de l'IA dans "
            "l'immobilier)."
        ),
        original_requirements_summary=(
            "Section standard et obligatoire. Doit fournir une clôture réflexive au "
            "mémoire."
        ),
        student_experience_keywords=[
            "synthèse des résultats",
            "réponse à la problématique IA",
            "bilan de l'alternance",
            "limites des projets menés",
            "perspectives d'évolution IA chez GECINA",
            "développement professionnel personnel",
            "ouverture sur futurs travaux",
            "leçons apprises",
        ],
        example_phrasing_or_content_type=(
            "Réflexif, synthétique, prospectif. Doit démontrer la prise de recul."
        ),
        key_questions_to_answer=[
            "Quelles sont les conclusions principales de mon travail ?",
            "Ma problématique a-t-elle trouvé une réponse ?",
            "Quelles sont les limites et les ouvertures possibles ?",
        ],
    ),
    PlannedSectionDetailForLLM(
        id="6.",
        title="Bibliographie",
        level=1,
        description_objectives=("Lister toutes les sources citées dans le mémoire."),
        original_requirements_summary=("Standard académique. Format à respecter."),
        student_experience_keywords=[
            "sources académiques",
            "articles de recherche IA",
            "documentation Epitech",
            "références professionnelles",
        ],
        example_phrasing_or_content_type=(
            "Liste formatée selon les normes (ex: APA, IEEE)."
        ),
        key_questions_to_answer=[
            "Toutes les sources sont-elles listées ?",
            "Le format est-il correct ?",
        ],
    ),
    PlannedSectionDetailForLLM(
        id="7.",
        title="Annexes",
        level=1,
        description_objectives=(
            "Fournir des documents complémentaires pertinents non inclus dans le corps "
            "principal."
        ),
        original_requirements_summary=(
            "Optionnel, mais utile pour des éléments volumineux ou de support."
        ),
        student_experience_keywords=[
            "schémas d'architecture détaillés",
            "extraits de code pertinents",
            "verbatims d'entretiens anonymisés",
            "glossaire technique",
        ],
        example_phrasing_or_content_type=(
            "Documents de support, organisés et référencés."
        ),
        key_questions_to_answer=[
            ("Quels documents appuient mon mémoire sans l'alourdir ?"),
        ],
    ),
]
REALISTIC_MOCK_LLM_OUTPUT = PlannedThesisOutlineForLLM(
    outline=REALISTIC_MOCK_PLANNED_SECTIONS
)


class TestN3ThesisOutlinePlannerNode(unittest.TestCase):
    def setUp(self):
        self.sample_guidelines: dict[str, list[str]] = {
            "INTRODUCTION GÉNÉRALE": ["Objectif", "Problématique"],
            "CHAPITRE 3 : ANALYSE DES COMPÉTENCES (RNCP35284)": [
                "Relier les expériences aux blocs de compétences RNCP.",
                "Bloc A1.1 : Recueillir, analyser et modéliser les besoins.",
            ],
        }
        self.sample_persona: str = (
            "Étudiant en Master IA, alternant chef de projet IA dans une foncière."
        )
        self.initial_state = AgentState(
            school_guidelines_structured=self.sample_guidelines,
            user_persona=self.sample_persona,
            example_thesis_text_content=EXAMPLE_THESIS_DIGI5_ABRIDGED_FOR_TEST,
        )

    @patch("src.nodes.n3_thesis_outline_planner.ChatOllama")
    def test_generates_outline_with_structured_output_success(
        self, mock_chat_ollama_class: MagicMock
    ):
        mock_llm_instance = mock_chat_ollama_class.return_value
        mock_structured_llm_instance = MagicMock()
        mock_llm_instance.with_structured_output.return_value = (
            mock_structured_llm_instance
        )
        mock_structured_llm_instance.invoke.return_value = REALISTIC_MOCK_LLM_OUTPUT

        planner_node_for_test = N3ThesisOutlinePlannerNode(
            llm_model_name="mock_model_for_structured_test", temperature=0.0
        )
        planner_node_for_test.use_fallback_parser = False

        updated_state_dict = planner_node_for_test.run(self.initial_state)

        assert "thesis_outline" in updated_state_dict
        outline: list[SectionDetail] = updated_state_dict["thesis_outline"]
        assert len(outline) == len(REALISTIC_MOCK_PLANNED_SECTIONS)

        for i, planned_section in enumerate(REALISTIC_MOCK_PLANNED_SECTIONS):
            state_section = outline[i]
            assert state_section.id == planned_section.id
            assert state_section.title == planned_section.title
            assert state_section.level == planned_section.level
            assert (
                state_section.description_objectives
                == planned_section.description_objectives
            )
            assert (
                state_section.original_requirements_summary
                == planned_section.original_requirements_summary
            )
            assert (
                state_section.student_experience_keywords
                == planned_section.student_experience_keywords
            )
            assert (
                state_section.example_phrasing_or_content_type
                == planned_section.example_phrasing_or_content_type
            )
            assert (
                state_section.key_questions_to_answer
                == planned_section.key_questions_to_answer
            )
            assert state_section.status == SectionStatus.PENDING

        assert updated_state_dict.get("error_message") is None
        assert (
            updated_state_dict["last_successful_node"] == "N3ThesisOutlinePlannerNode"
        )
        mock_llm_instance.with_structured_output.assert_called_once_with(
            schema=PlannedThesisOutlineForLLM
        )
        mock_structured_llm_instance.invoke.assert_called_once()

    @patch("src.nodes.n3_thesis_outline_planner.ChatOllama")
    def test_generates_outline_with_fallback_parser_success(
        self, mock_chat_ollama_class: MagicMock
    ):
        mock_llm_instance = mock_chat_ollama_class.return_value
        mock_ai_message = AIMessage(content=REALISTIC_MOCK_LLM_OUTPUT.json())
        mock_llm_instance.invoke.return_value = mock_ai_message

        planner_node_for_test = N3ThesisOutlinePlannerNode(
            llm_model_name="mock_model_for_fallback_test", temperature=0.0
        )
        planner_node_for_test.llm = mock_llm_instance
        planner_node_for_test.structured_llm = None
        planner_node_for_test.use_fallback_parser = True

        updated_state_dict = planner_node_for_test.run(self.initial_state)

        assert "thesis_outline" in updated_state_dict
        outline: list[SectionDetail] = updated_state_dict["thesis_outline"]
        assert len(outline) == len(REALISTIC_MOCK_PLANNED_SECTIONS)

        for i, planned_section in enumerate(REALISTIC_MOCK_PLANNED_SECTIONS):
            state_section = outline[i]
            assert state_section.id == planned_section.id
            assert state_section.title == planned_section.title
            assert state_section.level == planned_section.level
            assert (
                state_section.description_objectives
                == planned_section.description_objectives
            )
            assert (
                state_section.original_requirements_summary
                == planned_section.original_requirements_summary
            )
            assert (
                state_section.student_experience_keywords
                == planned_section.student_experience_keywords
            )
            assert (
                state_section.example_phrasing_or_content_type
                == planned_section.example_phrasing_or_content_type
            )
            assert (
                state_section.key_questions_to_answer
                == planned_section.key_questions_to_answer
            )
            assert state_section.status == SectionStatus.PENDING

        assert updated_state_dict.get("error_message") is None
        assert (
            updated_state_dict["last_successful_node"] == "N3ThesisOutlinePlannerNode"
        )
        mock_llm_instance.invoke.assert_called_once()

    @patch("src.nodes.n3_thesis_outline_planner.ChatOllama")
    def test_handles_general_llm_invocation_error(
        self, mock_chat_ollama_class: MagicMock
    ):
        mock_llm_instance = mock_chat_ollama_class.return_value
        mock_llm_instance.invoke.side_effect = Exception("Simulated LLM API error")

        planner_node_for_test = N3ThesisOutlinePlannerNode(
            llm_model_name="mock_error_test"
        )
        planner_node_for_test.llm = mock_llm_instance
        planner_node_for_test.use_fallback_parser = True

        updated_state_dict = planner_node_for_test.run(self.initial_state)
        assert "error_message" in updated_state_dict
        assert "Simulated LLM API error" in updated_state_dict["error_message"]
        assert len(updated_state_dict["thesis_outline"]) == 1
        assert updated_state_dict["thesis_outline"][0].status == SectionStatus.ERROR

    @patch("src.nodes.n3_thesis_outline_planner.ChatOllama")
    def test_handles_pydantic_validation_error_in_fallback(
        self, mock_chat_ollama_class: MagicMock
    ):
        mock_llm_instance = mock_chat_ollama_class.return_value
        incomplete_json_content = '{"outline": [{"title": "Incomplete Section"}]}'
        mock_ai_message = AIMessage(content=incomplete_json_content)
        mock_llm_instance.invoke.return_value = mock_ai_message

        planner_node_for_test = N3ThesisOutlinePlannerNode(
            llm_model_name="mock_pyd_error_test"
        )
        planner_node_for_test.llm = mock_llm_instance
        planner_node_for_test.use_fallback_parser = True

        updated_state_dict = planner_node_for_test.run(self.initial_state)
        assert "error_message" in updated_state_dict
        assert "N3 Erreur Pydantic" in updated_state_dict["error_message"]
        assert "field required" in updated_state_dict["error_message"]
        assert len(updated_state_dict["thesis_outline"]) == 1
        assert updated_state_dict["thesis_outline"][0].status == SectionStatus.ERROR


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
