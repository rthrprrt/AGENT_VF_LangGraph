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
Gecina est une grande foncière. Description de son activité, son positionnement
sur le marché immobilier, et sa stratégie de développement durable CAN0P-2030.
Mention de ses services comme Yourplace et FEAT.
### 1.2. Ma mission et mon rôle
Ma mission était de développer l'IA chez Gecina. J'étais AIPO, rattaché à
Jérôme Carecchio. J'ai travaillé sur le prompt engineering pour les RH, et sur
une solution Power Automate pour la DG.
## CHAPITRE 3 : ANALYSE DES COMPÉTENCES (RNCP35284)
### 3.1. Bloc 1: Manager un projet Informatique et Systèmes d'Information
#### 3.1.1. Activité : Analyser les besoins des utilisateurs
Contenu: Projet X - Analyse des besoins clients pour un chatbot. Projet Alpha,
analyse des besoins pour la Direction Financière, utilisation de Power BI pour
des dashboards. J'ai identifié les KPIs.
#### 3.1.2. Activité : Concevoir la solution technique
Contenu: Projet Y - Conception de l'architecture d'un outil de prédiction.
Solution Power Automate pour la DG: architecture, OCR, AI Builder, fonction Azure.
...
"""

# FINALIZED REALISTIC_MOCK_PLANNED_SECTIONS (Reflecting final prompt output)
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
            "collaboration projet Automatisation DG",
            "opportunité d'alternance GECINA",
            "mentorat Jérôme Carecchio",
            "apprentissage gestion de projet IA",
        ],
        example_phrasing_or_content_type=(
            "Ton personnel et professionnel, sincère. Courte introduction et liste "
            "de remerciements nominatifs ou par groupe, similaire à Digi5."
        ),
        key_questions_to_answer=[
            "Qui a joué un rôle déterminant dans mon parcours et la réalisation de ce mémoire ?",
            "Comment exprimer ma gratitude de manière formelle et sincère, en reconnaissant les apports spécifiques ?",
        ],
    ),
    PlannedSectionDetailForLLM(
        id="0.2.",
        title="Sommaire Détaillé",
        level=1,
        description_objectives=(
            "Fournir une table des matières claire, détaillée et hiérarchisée du mémoire, "
            "permettant une navigation aisée et une compréhension rapide de la structure "
            "globale du document."
        ),
        original_requirements_summary=(
            "Section standard et indispensable pour tout document académique long. "
            "Doit refléter fidèlement la structure finale du mémoire, incluant tous "
            "les niveaux de titres."
        ),
        student_experience_keywords=[
            "élaboration du plan détaillé du mémoire",
            "structuration logique des chapitres et sections",
            "organisation hiérarchique du contenu",
            "vérification de la cohérence du plan",
            "préparation de la table des matières finale",
            "numérotation des sections",
        ],
        example_phrasing_or_content_type=(
            "Liste hiérarchique des titres de chapitres, sections et sous-sections avec "
            "numérotation conforme à l'exemple Digi5 (ex: 1., 1.1., 1.1.1.). La "
            "pagination sera ajoutée automatiquement à la fin."
        ),
        key_questions_to_answer=[
            "Quelles sont toutes les parties principales, secondaires et tertiaires de mon mémoire ?",
            "La numérotation et la hiérarchie des sections sont-elles logiques, cohérentes et conformes aux standards académiques et à l'exemple Digi5 ?",
        ],
    ),
    PlannedSectionDetailForLLM(
        id="1.",
        title="Introduction Générale",
        level=1,
        description_objectives=(
            "Présenter le contexte général de la mission d'alternance au sein de "
            "GECINA, introduire la problématique centrale du mémoire liée à "
            "l'intégration et à la gestion de projets d'Intelligence Artificielle dans "
            "le secteur spécifique de la foncière immobilière, définir les objectifs "
            "visés (notamment en lien avec le titre RNCP 35284), et annoncer de manière "
            "claire le plan du document."
        ),
        original_requirements_summary=(
            "Les directives Epitech insistent sur la clarté de la problématique et des "
            "objectifs. L'exemple Digi5 montre une introduction structurée (contexte, "
            "problématique, objectifs, annonce du plan)."
        ),
        student_experience_keywords=[
            "contexte de l'alternance chez GECINA en tant qu'AIPO",
            "problématique de l'IA dans le secteur foncier",
            "objectifs du mémoire en lien avec le RNCP 35284",
            "annonce de la structure du mémoire",
            "enjeux de la transformation digitale pour GECINA",
            "mon rôle de Chef de Projet IA (AIPO)",
            "opportunités et défis de l'IA pour GECINA",
            "contribution attendue du mémoire",
        ],
        example_phrasing_or_content_type=(
            "Style formel, académique, clair et concis. Doit être engageant, "
            "contextualiser le sujet et justifier l'importance de la problématique "
            "traitée."
        ),
        key_questions_to_answer=[
            "Quel était le contexte spécifique de ma mission chez GECINA et son lien avec l'IA ?",
            "Quelle problématique centrale ce mémoire aborde-t-il concernant l'IA dans le secteur de l'immobilier d'entreprise ?",
            "Quels sont les objectifs de ce mémoire et quelle structure vais-je suivre pour y répondre et démontrer mes compétences RNCP ?",
        ],
    ),
    PlannedSectionDetailForLLM(
        id="1.1.",
        title="Description de l'entreprise d'accueil : GECINA",
        level=2,
        description_objectives=(
            "Présenter en détail GECINA : son statut de Société d'Investissement "
            "Immobilier Cotée (SIIC), son secteur d'activité principal (immobilier de "
            "bureaux), son positionnement stratégique sur le marché (notamment focus "
            "Paris et grands pôles tertiaires), sa démarche RSE ambitieuse (ex: "
            "stratégie CAN0P-2030) et ses offres de services innovants (ex: Yourplace, "
            "FEAT)."
        ),
        original_requirements_summary=(
            "Déduit de l'exemple Digi5 et des directives Epitech qui demandent de "
            "contextualiser la mission professionnelle. Cette section est cruciale "
            "pour comprendre l'environnement dans lequel l'alternance s'est déroulée."
        ),
        student_experience_keywords=[
            "analyse sectorielle GECINA",
            "compréhension du modèle économique SIIC",
            "étude du marché de l'immobilier de bureaux à Paris",
            "contribution à la stratégie RSE CAN0P-2030 de GECINA",
            "veille concurrentielle et services innovants Yourplace et FEAT",
            "découverte de l'organisation et des départements de GECINA",
            "intégration à la culture d'entreprise et valeurs de GECINA",
            "évaluation du portefeuille d'actifs immobiliers de GECINA",
            "analyse SWOT de GECINA (forces, faiblesses, opportunités, menaces)",
        ],
        example_phrasing_or_content_type=(
            "Contenu factuel, analytique, précis et si possible, illustré par des "
            "données chiffrées issues de sources publiques (rapports annuels, site web "
            "de GECINA). S'inspirer du style descriptif et informatif de Digi5 pour "
            "cette section."
        ),
        key_questions_to_answer=[
            "Qu'est-ce que GECINA : son activité principale, son statut juridique et financier, et sa taille ?",
            "Comment GECINA se positionne-t-elle sur son marché concurrentiel et quelle est sa stratégie distinctive en matière de RSE et d'innovation technologique ?",
        ],
    ),
    PlannedSectionDetailForLLM(
        id="1.2.",
        title=(
            "Ma mission en tant que Chef de Projet IA (AIPO) et objectifs du " "mémoire"
        ),
        level=2,
        description_objectives=(
            "Décrire en détail ma mission spécifique en tant qu'AIPO (Artificial "
            "Intelligence Project Officer) au sein de GECINA, en mettant en lumière "
            "les projets IA principaux sur lesquels j'ai activement travaillé (ex: "
            "'Projet Prompt Engineering pour les RH', 'Projet d'Automatisation des "
            "processus pour la Direction Générale via Power Automate'), mon "
            "positionnement hiérarchique et fonctionnel (rattachement à Jérôme "
            "Carecchio, interactions avec les équipes). Clarifier les objectifs "
            "spécifiques de ce mémoire en lien direct avec le titre RNCP 35284 et "
            "l'auto-évaluation de mes compétences acquises et mobilisées."
        ),
        original_requirements_summary=(
            "Les directives Epitech exigent de lier de manière explicite le mémoire "
            "à la mission professionnelle et aux compétences du référentiel RNCP. "
            "L'exemple Digi5 détaille la mission, son contexte et les projets menés."
        ),
        student_experience_keywords=[
            "définition du rôle AIPO chez GECINA",
            "pilotage de projets IA concrets",
            "développement de prompts pour le service RH (Projet Prompt RH)",
            "conception et développement solution Power Automate (Projet Automatisation DG)",
            "utilisation technologies OCR, AI Builder, Azure Functions",
            "collaboration et reporting à mon tuteur Jérôme Carecchio",
            "alignement de ma mission avec les blocs de compétences RNCP 35284",
            "identification des objectifs de développement de compétences",
            "auto-évaluation de mes performances en tant qu'AIPO",
            "contribution à la feuille de route IA de GECINA",
        ],
        example_phrasing_or_content_type=(
            "Style descriptif, analytique et professionnel. Mettre en avant les "
            "responsabilités assumées, les technologies spécifiques utilisées, les "
            "méthodologies de projet appliquées et les enjeux stratégiques des projets "
            "menés. Énoncer clairement les buts du mémoire et sa contribution à la "
            "validation du titre RNCP."
        ),
        key_questions_to_answer=[
            "Quelle était ma mission principale et mes responsabilités concrètes en tant "
            "qu'AIPO chez GECINA, et sur quels projets IA spécifiques ai-je travaillé ?",
            "Quels sont les objectifs de ce mémoire, notamment par rapport au référentiel "
            "RNCP 35284 et à mon développement personnel et professionnel de compétences en IA ?",
        ],
    ),
    PlannedSectionDetailForLLM(
        id="3.",
        title="Analyse des Compétences (RNCP 35284)",
        level=1,
        description_objectives=(
            "Analyser de manière approfondie et structurée, bloc par bloc, les "
            "compétences du référentiel RNCP 35284 (Expert en management des systèmes "
            "d'information) que j'ai développées, acquises ou mobilisées durant mon "
            "alternance chez GECINA. Fournir des exemples concrets et détaillés issus de "
            "mes projets ('Projet Prompt Engineering RH', 'Projet Automatisation DG', "
            "'Projet Power BI Finance') et de mes tâches quotidiennes pour chaque "
            "compétence ou activité clé du référentiel."
        ),
        original_requirements_summary=(
            "Section absolument cruciale et centrale du mémoire selon les directives "
            "Epitech. Doit couvrir de manière exhaustive les blocs de compétences RNCP "
            "(Bloc 1: Manager un projet Informatique et SI; Bloc 2: Définir et déployer "
            "la stratégie SI; Bloc 3: Piloter la performance et la sécurité du SI; "
            "Bloc 4: Manager les équipes SI; Bloc 5: Accompagner la transformation "
            "digitale). L'exemple Digi5 adopte une structure claire par bloc et par "
            "activité."
        ),
        student_experience_keywords=[
            "mise en application des compétences RNCP 35284",
            "analyse des besoins utilisateurs pour Projet Automatisation DG",
            "conception de la solution Power Automate pour DG",
            "gestion de projet d'automatisation de processus financiers",
            "maintenance évolutive et corrective du chatbot RH",
            "création de documentation et support utilisateurs pour outil Power BI",
            "veille technologique sur l'IA générative et son application à l'immobilier",
            "présentation de preuves de compétence via projets concrets",
            "analyse des difficultés rencontrées et des solutions innovantes apportées",
            "contributions spécifiques à l'amélioration de la performance des SI chez GECINA",
            "gestion des parties prenantes dans les projets IA",
        ],
        example_phrasing_or_content_type=(
            "Structurer impérativement par bloc de compétences RNCP, puis par activité "
            "clé au sein de chaque bloc, en respectant la nomenclature officielle et en "
            "s'inspirant de la présentation de Digi5. Pour chaque activité, utiliser une "
            "méthode descriptive et analytique (ex: STAR - Situation, Tâche, Action, "
            "Résultat; ou narration de projet détaillée) pour illustrer avec des exemples "
            "précis et chiffrés si possible."
        ),
        key_questions_to_answer=[
            "Quelles compétences spécifiques de chaque bloc du référentiel RNCP 35284 "
            "ai-je mises en œuvre et développées au cours de mon alternance chez GECINA ?",
            "Comment mes expériences professionnelles (projets, tâches spécifiques, "
            "défis relevés) illustrent-elles concrètement l'acquisition, l'application et "
            "la maîtrise de ces compétences ?",
            "Quels ont été les résultats tangibles de mes actions et les apprentissages "
            "clés pour chaque compétence mobilisée, en lien avec les attendus du RNCP ?",
        ],
    ),
    PlannedSectionDetailForLLM(
        id="3.1.",
        title="Bloc 1: Manager un projet Informatique et Systèmes d'Information",
        level=2,
        description_objectives=(
            "Détailler les activités et compétences relevant du Bloc 1 du RNCP35284, "
            "notamment 'C1.1 Analyser les besoins des utilisateurs et des organisations "
            "en matière de SI' et 'C1.2 Concevoir la solution technique et "
            "fonctionnelle d'un projet SI'. Illustrer avec les projets d'IA spécifiques "
            "menés chez GECINA, comme le 'Projet Automatisation DG' et le 'Projet "
            "Prompt Engineering RH'."
        ),
        original_requirements_summary=(
            "RNCP35284 - Bloc 1. L'exemple Digi5 structure ce bloc par activités "
            "spécifiques (Analyser les besoins, Concevoir la solution). Mettre "
            "l'accent sur la dimension managériale et la conduite de projet SI."
        ),
        student_experience_keywords=[
            "management de projet SI chez GECINA",
            "analyse des besoins utilisateurs pour Projet Automatisation DG",
            "rédaction de spécifications fonctionnelles pour Projet Prompt RH",
            "conception de l'architecture technique de la solution Power Automate",
            "étude de faisabilité technique et économique projet IA",
            "sélection des technologies (Power Platform, Azure AI Services, modèles LLM)",
            "application de méthodologies de gestion de projet (ex: Agile, Scrum-light)",
            "planification des phases et des livrables projet IA",
            "identification et gestion des risques projet SI",
            "communication et coordination avec les parties prenantes métiers et IT",
            "estimation des charges et des coûts projet",
        ],
        example_phrasing_or_content_type=(
            "Présenter chaque activité clé du bloc (C1.1, C1.2, etc.). Pour chaque "
            "activité, décrire les projets spécifiques où elle a été mise en œuvre, en "
            "détaillant les actions managériales, les outils de gestion de projet (ex: "
            "Jira, Trello, MS Project), les méthodes (ex: entretiens, ateliers, MoSCoW) "
            "et les résultats obtenus en termes de pilotage et de livraison de solution."
        ),
        key_questions_to_answer=[
            "Comment ai-je analysé les besoins des utilisateurs et de GECINA pour les "
            "projets SI que j'ai managés ou auxquels j'ai contribué activement ?",
            "Quelles solutions techniques et fonctionnelles ai-je conçues, et comment "
            "ai-je assuré leur adéquation avec les besoins exprimés et les contraintes "
            "du projet (budget, délais, ressources) ?",
            "Comment ai-je planifié et suivi l'avancement de ces projets SI ?",
        ],
    ),
    PlannedSectionDetailForLLM(
        id="3.1.1.",
        title=(
            "Activité C1.1 : Analyser les besoins des utilisateurs et des organisations "
            "en matière de Systèmes d'Information"
        ),
        level=3,
        description_objectives=(
            "Illustrer par des exemples concrets et précis (ex: 'Projet Prompt "
            "Engineering RH', 'Projet Automatisation DG', 'Projet Power BI Finance') "
            "comment j'ai recueilli, évalué, formalisé et modélisé les besoins des "
            "utilisateurs (ex: Direction Financière, département RH, Direction Générale) "
            "et de l'organisation GECINA pour des projets de Systèmes d'Information "
            "basés sur l'IA."
        ),
        original_requirements_summary=(
            "Déduit de la structure de l'exemple Digi5 (sous-sections pour chaque "
            "activité du bloc RNCP) et du contenu spécifique de l'activité C1.1 du "
            "Bloc 1 du RNCP. Met l'accent sur les techniques et livrables de l'analyse "
            "des besoins."
        ),
        student_experience_keywords=[
            "animation d'ateliers de recueil des besoins pour Projet Prompt RH",
            "réalisation d'entretiens utilisateurs avec la Direction Générale pour "
            "le Projet Automatisation DG",
            "définition des Key Performance Indicators (KPIs) pour le reporting "
            "financier (Projet Power BI Finance)",
            "formalisation des exigences métier pour le chatbot RH",
            (
                "rédaction de spécifications fonctionnelles détaillées (SFD) pour la "
                "solution Power Automate DG"
            ),
            "analyse des points de douleur (pain points) des processus manuels existants",
            (
                "construction et priorisation du backlog produit pour le Projet d'IA "
                "de classification de documents"
            ),
            "réalisation d'une étude d'opportunité pour une solution d'IA dans le "
            "département Juridique",
            "cartographie des processus métier existants avant leur automatisation (BPMN)",
            "validation des besoins et des spécifications avec les parties prenantes "
            "métiers et techniques",
        ],
        example_phrasing_or_content_type=(
            "Décrire un ou plusieurs projets spécifiques (ex: 'Projet Prompt "
            "Engineering RH', 'Projet Automatisation DG'). Expliquer en détail les "
            "méthodes et outils utilisés pour l'analyse des besoins (ex: ateliers de "
            "co-conception, entretiens individuels structurés, analyse documentaire, "
            "mind mapping, diagrammes de flux BPMN pour la modélisation de processus) et "
            "les livrables produits (ex: comptes-rendus d'ateliers, spécifications "
            "fonctionnelles, user stories, maquettes fonctionnelles, diagrammes de "
            "processus 'as-is' et 'to-be')."
        ),
        key_questions_to_answer=[
            (
                "Quels étaient les besoins utilisateurs et organisationnels spécifiques "
                "identifiés pour le projet [Nom du Projet, ex: Automatisation DG] et "
                "comment se traduisaient-ils en exigences SI ?"
            ),
            (
                "Comment ai-je procédé concrètement (méthodes, outils) pour recueillir, "
                "analyser, modéliser et valider ces besoins auprès des différentes parties "
                "prenantes ?"
            ),
            (
                "Quels ont été les livrables clés de cette phase d'analyse (ex: SFD, "
                "backlog) et comment ont-ils orienté la conception et le développement de "
                "la solution SI ?"
            ),
        ],
    ),
    PlannedSectionDetailForLLM(
        id="5.",
        title="Conclusion Générale",
        level=1,
        description_objectives=(
            "Synthétiser les principaux apports et résultats du mémoire, répondre de "
            "manière concise et argumentée à la problématique posée en introduction, "
            "discuter des limites du travail réalisé et des projets menés durant "
            "l'alternance. Proposer des perspectives d'évolution futures tant sur le "
            "plan personnel (développement continu de compétences) que pour l'entreprise "
            "GECINA (potentialités et recommandations pour l'IA) et pour le domaine plus "
            "large de l'IA dans le secteur de l'immobilier d'entreprise."
        ),
        original_requirements_summary=(
            "Section standard et obligatoire de tout mémoire académique. Doit fournir "
            "une clôture réflexive, synthétique et prospective au document, en revenant "
            "sur les objectifs initiaux et en évaluant leur atteinte."
        ),
        student_experience_keywords=[
            "synthèse des contributions et résultats clés du mémoire",
            "réponse argumentée à la problématique de l'IA dans l'immobilier",
            "bilan critique et constructif de l'expérience d'alternance chez GECINA",
            "identification des limites méthodologiques ou techniques des projets IA menés",
            "recommandations stratégiques pour l'évolution de l'IA chez GECINA",
            "plan de développement professionnel personnel et objectifs futurs",
            (
                "ouverture sur les futures tendances et innovations en IA appliquées au "
                "secteur foncier"
            ),
            "leçons apprises en gestion de projet IA et en collaboration d'équipe",
            "évaluation de l'impact concret de ma mission et de mes projets",
            "réflexion sur les aspects éthiques et sociétaux de l'IA mise en œuvre",
        ],
        example_phrasing_or_content_type=(
            "Style réflexif, synthétique et prospectif. Doit démontrer une prise de "
            "recul critique sur le travail accompli, une capacité à identifier les "
            "apprentissages significatifs et une vision pour l'avenir. Conclure sur une "
            "note positive et constructive."
        ),
        key_questions_to_answer=[
            (
                "Quelles sont les conclusions principales de mon travail de mémoire et de "
                "mon expérience d'alternance chez GECINA en tant qu'AIPO ?"
            ),
            (
                "Ma problématique initiale concernant l'intégration et la gestion de "
                "projets IA dans le secteur immobilier a-t-elle trouvé une réponse "
                "satisfaisante et nuancée à travers mes analyses et réalisations ?"
            ),
            (
                "Quelles sont les limites de mes réalisations et de mes analyses, et quelles "
                "perspectives d'amélioration, d'évolution ou de recherche future puis-je "
                "envisager pour moi-même, pour GECINA et pour le domaine ?"
            ),
        ],
    ),
    PlannedSectionDetailForLLM(
        id="6.",
        title="Bibliographie",
        level=1,
        description_objectives=(
            "Lister de manière exhaustive, précise et formalisée toutes les sources "
            "documentaires (ouvrages scientifiques, articles académiques, articles de "
            "presse spécialisée, sites web de référence, documentations techniques, "
            "rapports d'entreprise, cours Epitech, etc.) qui ont été citées ou "
            "consultées de manière significative pour la rédaction du mémoire."
        ),
        original_requirements_summary=(
            "Section académique standard et obligatoire. Le format de citation doit être "
            "cohérent et respecter les normes attendues par Epitech (si spécifiées, "
            "sinon choisir un standard reconnu et l'appliquer rigoureusement, ex: APA, "
            "IEEE, Harvard)."
        ),
        student_experience_keywords=[
            "références d'articles scientifiques sur l'IA et le NLP",
            "sources de recherche sur le machine learning et l'automatisation",
            "documentation technique Microsoft Power Platform et Azure AI Services",
            "rapports annuels et publications de GECINA",
            "articles de presse sur la transformation digitale dans l'immobilier",
            "utilisation d'outils de gestion bibliographique (ex: Zotero, Mendeley)",
            "respect des normes de citation académique",
            "veille informationnelle et technologique continue durant l'alternance",
            "sources internes GECINA (anonymisées si nécessaire)",
            "cours et supports pédagogiques Epitech pertinents",
        ],
        example_phrasing_or_content_type=(
            "Liste ordonnée (généralement alphabétique par nom d'auteur principal) et "
            "formatée selon les normes de citation choisies (ex: auteur, année, titre, "
            "source). Doit être précise et permettre au lecteur de retrouver facilement "
            "chaque source mentionnée."
        ),
        key_questions_to_answer=[
            (
                "Toutes les sources utilisées explicitement (citations directes ou "
                "indirectes) ou implicitement (connaissances de fond) dans le mémoire "
                "sont-elles correctement et intégralement listées ici ?"
            ),
            (
                "Le format de citation est-il appliqué de manière rigoureuse, complète et "
                "cohérente pour toutes les références bibliographiques listées ?"
            ),
        ],
    ),
    PlannedSectionDetailForLLM(
        id="7.",
        title="Annexes",
        level=1,
        description_objectives=(
            "Fournir des documents complémentaires jugés pertinents qui appuient le "
            "contenu du mémoire mais qui ne sont pas essentiels à sa compréhension "
            "directe ou qui seraient trop volumineux ou trop techniques pour être "
            "inclus dans le corps principal du document (ex: schémas d'architecture "
            "technique détaillés, extraits significatifs de code source, verbatims "
            "complets d'entretiens anonymisés, glossaire technique exhaustif, tableaux "
            "de données brutes ou complémentaires)."
        ),
        original_requirements_summary=(
            "Section optionnelle, mais souvent très utile pour des éléments de support, "
            "des preuves détaillées, des illustrations techniques ou des compléments "
            "d'information qui enrichissent le mémoire. Doit être organisée, chaque "
            "annexe étant numérotée et titrée, et clairement référencée depuis le corps "
            "du texte si les annexes sont explicitement mentionnées."
        ),
        student_experience_keywords=[
            (
                "schémas d'architecture détaillés de la solution Power Automate pour la "
                "Direction Générale"
            ),
            "extraits de prompts complexes et itérations pour le service RH",
            "exemples de scripts Python pour le prétraitement de données IA",
            "verbatims anonymisés d'entretiens utilisateurs clés pour l'analyse des besoins",
            "glossaire des termes techniques IA et des acronymes spécifiques à GECINA",
            (
                "tableaux de données de performance des modèles IA avant/après "
                "optimisation"
            ),
            "captures d'écran illustratives d'interfaces utilisateur développées",
            "résultats détaillés d'enquêtes ou de questionnaires menés",
            "charte de projet IA (anonymisée)",
            "plan de communication projet (anonymisé)",
        ],
        example_phrasing_or_content_type=(
            "Collection de documents de support, clairement numérotés (Annexe 1, Annexe 2, "
            "etc.) et titrés de manière explicite. Chaque annexe doit être idéalement "
            "mentionnée au moins une fois dans le corps du mémoire (ex: 'Voir Annexe 3 "
            "pour le schéma d'architecture détaillé') pour justifier sa présence et guider "
            "le lecteur."
        ),
        key_questions_to_answer=[
            (
                "Quels documents complémentaires sont véritablement nécessaires pour étayer "
                "mes propos, fournir des détails techniques importants ou illustrer mes "
                "réalisations sans alourdir excessivement le corps du mémoire ?"
            ),
            (
                "Ces annexes sont-elles bien organisées, clairement titrées, numérotées et "
                "facilement compréhensibles par un lecteur souhaitant approfondir certains "
                "aspects ?"
            ),
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
        """Teste la génération de plan avec sortie structurée LLM."""
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
        """Teste la génération de plan avec le parser fallback."""
        mock_llm_instance = mock_chat_ollama_class.return_value
        # Correction: Utiliser .json() pour Pydantic V1
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
            # ... (autres assertions pour chaque champ)
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
        """Teste la gestion d'erreur générale lors de l'appel LLM."""
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
        """Teste la gestion d'erreur de validation Pydantic en mode fallback."""
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


if __name__ == "__main__":  # pragma: no cover
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
