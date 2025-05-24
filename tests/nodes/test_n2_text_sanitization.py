# tests/nodes/test_n2_text_sanitization.py
import logging
import re
from datetime import date
from typing import Any

import pytest
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


# --- Début de la copie de TextSanitizationExpert et ses modèles internes ---
class RawJournalEntryForTest(BaseModel):
    # Ligne 9 (E501) - déjà OK, potentiellement un reformatage par Ruff l'a remise
    text: str
    source_file: str
    entry_date: date | None = None


class ProcessedJournalEntryForTest(RawJournalEntryForTest):
    anonymized_text: str
    tone_issues: list[dict[str, str]] = Field(default_factory=list)
    requires_tone_review: bool = False
    llm_rephrased_passages: list[dict[str, str]] = Field(default_factory=list)


class AppSettingsForTest:
    llm_model_name: str = "gemma2:9b"


class TextSanitizationExpert:
    """Encapsulates logic for text anonymization and tone management."""

    def __init__(
        self,
        initial_anonymization_map: dict[str, str] | None = None,
        settings: AppSettingsForTest | None = None,
    ):
        self.settings = settings if settings else AppSettingsForTest()
        self.anonymization_map = (
            initial_anonymization_map.copy() if initial_anonymization_map else {}
        )

        self.org_chart_map_config = {
            "Beñat Ortega": "[Le_Directeur_General]",
            "Nicolas Dutreuil": "[Le_Directeur_General_Adjoint_Finances]",
            "Nicolas Broband": "[Le_Directeur_Communication_Financiere]",
            "Thierry Perisser": "[Le_DSI]",
            "TP": "[Le_DSI]",
            "Jérôme Carecchio": "[Mon_Responsable_Projets_Info]",
            "Jérôme": "[Mon_Responsable_Projets_Info]",
            "Romain Hardy": "[Le_Directeur_Corporate_Finance]",
            "Agnès Arnaud": "[L_Assistante_Direction]",
            "Valérie Britay": "[La_Directrice_Adjointe_Pole_Bureau]",
            "Marie Lalande": "[La_Directrice_Executive_Ingenierie_RSE]",
            "Brahim Annour": "[Le_Directeur_Innovation]",
            "Brahim": "[Le_Directeur_Innovation]",
            "Alexandre Morel": "[Le_Chef_Projet_Innovation]",
            "Souhail Ouakkas": "[Le_Stagiaire_IA_Innovation]",
            "Romain Veber": "[Le_Directeur_Executif_Investissements_Developpement]",
            "Pierre-emmanuel Bandioli": "[Le_Directeur_Executif_Pole_Residentiel]",
            "Christine Harné": "[La_Directrice_Executive_RH]",
        }
        for key, value in self.org_chart_map_config.items():
            if key not in self.anonymization_map:
                self.anonymization_map[key] = value
            elif self.anonymization_map[key] != value and not self.anonymization_map[
                key
            ].startswith(("[Person_", "[EC_", "[IP_")):
                log_msg = (
                    f"Clash pour '{key}'. Priorité organigramme: '{value}' vs "
                    f"'{self.anonymization_map[key]}'"
                )
                logger.info(log_msg)
                self.anonymization_map[key] = value

        self.sorted_org_chart_keys = sorted(
            self.org_chart_map_config.keys(), key=len, reverse=True
        )
        self.other_known_entities_config = {
            "Magic Lemp": "[External_Vendor_A]",
            "Expertime": "[External_Vendor_B]",
        }
        self.sorted_other_known_keys = sorted(
            self.other_known_entities_config.keys(), key=len, reverse=True
        )
        self.whitelisted_projects = ["Héraclès"]
        self.author_name = "Arthur Perrot"
        self.host_company_name = "Gecina"
        self.counters = self._initialize_counters_from_map(self.anonymization_map)
        self.regex_patterns = {
            "project": re.compile(
                r"\b(Projet|Project)\s+([A-Za-z0-9\-\_À-ÿ]+)\b", re.IGNORECASE
            ),
            "common_first_names": re.compile(
                r"\b(Yann|Nicolas|Benat)(?!\w)\b", re.IGNORECASE
            ),
        }
        self.common_first_names_list = ["Yann", "Benat", "Nicolas"]
        self.tone_keywords_subjective = [
            "je pense que",
            "à mon avis",
            "il me semble",
            "je crois",
            "selon moi",
            "personnellement",
        ]
        self.tone_keywords_negative_emotion = [
            "frustré",
            "déteste",
            "agacé",
            "terrible",
            "nul",
            "problème",
            "pire",
            "mauvais",
            "énerve",
            "ennuie",
            "regrette",
            "déçu",
        ]
        self.tone_keywords_strong_doubt = [
            "ne me rassure pas",
            "pas sûr de",
            "pas certain",
            "incertain",
            "douteux",
            "sceptique",
        ]
        self.tone_keywords_informal = [
            "truc",
            "machin",
            "super",
            "génial",
            "cool",
            "boulot",
            "taf",
        ]
        self.regex_personal_opinion = re.compile(
            r"\b(?:Je|J['’])\s+(?!suis\b|ai\b|vois\b|dois\b|peux\b|vais\b|sais\b"
            r"|travaille\b|fais\b|commence\b|termine\b|participe\b|présente\b"
            r"|note\b|comprends\b|réalise\b)([a-zA-Zà-ÿ']+).*\b",
            re.IGNORECASE,
        )

    def _int_to_char_suffix(self, n: int) -> str:
        """Converts a 0-indexed integer to A, B,... Z, AA, AB..."""
        if n < 0:
            return ""
        string = ""
        temp_n = n
        while True:
            temp_n, remainder = divmod(temp_n, 26)
            string = chr(ord("A") + remainder) + string
            if temp_n == 0:
                break
            temp_n -= 1
        return string

    def _char_suffix_to_int(self, s: str) -> int:
        """Converts A=0, B=1,... Z=25, AA=26... to 0-indexed integer."""
        val_corrected = -1
        for char_val in s:
            val_corrected = (val_corrected + 1) * 26 + (ord(char_val) - ord("A"))
        return val_corrected

    def _initialize_counters_from_map(
        self, anonymization_map: dict[str, str]
    ) -> dict[str, int]:
        """Initializes placeholder counters based on the current map."""
        counters_initialized = {"person": -1, "company": -1, "project": -1}
        patterns = {
            "person": re.compile(r"\[Person_([A-Z]+)\]"),
            "company": re.compile(r"\[External_Company_([A-Z]+)\]"),
            "project": re.compile(r"\[Internal_Project_([A-Z]+)\]"),
        }
        for entity_type, pattern in patterns.items():
            current_max_suffix = -1
            for placeholder_value in anonymization_map.values():
                match = pattern.fullmatch(placeholder_value)
                if match:
                    try:
                        val = self._char_suffix_to_int(match.group(1))
                        current_max_suffix = max(current_max_suffix, val)
                    except ValueError:
                        logger.warning(
                            "Suffixe placeholder invalide: %s", match.group(1)
                        )
            counters_initialized[entity_type] = current_max_suffix
        return counters_initialized

    def _generate_placeholder(self, entity_type: str) -> str:
        """Generates a new placeholder, incrementing instance counters."""
        self.counters[entity_type] += 1
        suffix = self._int_to_char_suffix(self.counters[entity_type])
        if entity_type == "person":
            return f"[Person_{suffix}]"
        if entity_type == "company":
            return f"[External_Company_{suffix}]"
        if entity_type == "project":
            return f"[Internal_Project_{suffix}]"
        return f"[UNKNOWN_ENTITY_{suffix}]"

    def _apply_replacements_from_map(
        self, text: str, replacement_map: dict[str, str], sorted_keys: list[str]
    ) -> str:
        """Helper to apply replacements from a specific map to text."""
        for key in sorted_keys:
            placeholder = replacement_map.get(key)
            if not placeholder:
                continue
            if (
                key not in self.anonymization_map
                or self.anonymization_map[key] != placeholder
            ):
                self.anonymization_map[key] = placeholder
                logger.debug("Map (specific map): '%s' -> '%s'", key, placeholder)

            pattern = r"\b" + re.escape(key) + r"\b"
            text = re.sub(pattern, placeholder, text, flags=re.IGNORECASE)
        return text

    def _apply_regex_project_replacements_internal(self, text: str) -> str:
        """Helper for regex-based project replacements, updates instance map."""
        project_replacements: list[tuple[str, str]] = []
        project_regex = self.regex_patterns["project"]
        for match in project_regex.finditer(text):
            project_name = match.group(0)
            project_identifier = match.group(2) or project_name
            project_name_lower = project_name.lower()
            is_whitelisted = any(
                wl.lower() == project_identifier.lower()
                or wl.lower() == project_name_lower
                for wl in self.whitelisted_projects
            )
            if not is_whitelisted:
                placeholder = self.anonymization_map.get(project_name)
                if not placeholder:
                    placeholder = self._generate_placeholder("project")
                    self.anonymization_map[project_name] = placeholder
                project_replacements.append((project_name, placeholder))

        project_replacements.sort(key=lambda x: len(x[0]), reverse=True)
        for name, placeholder in project_replacements:
            pattern = r"\b" + re.escape(name) + r"\b"
            text = re.sub(pattern, placeholder, text, flags=re.IGNORECASE)
        return text

    def _anonymize_text_single_entry_main_logic(self, text: str) -> str:
        """Main logic for anonymizing a single text entry."""
        anonymized_text = text
        anonymized_text = self._apply_replacements_from_map(
            anonymized_text, self.org_chart_map_config, self.sorted_org_chart_keys
        )
        anonymized_text = self._apply_replacements_from_map(
            anonymized_text,
            self.other_known_entities_config,
            self.sorted_other_known_keys,
        )
        anonymized_text = self._apply_regex_project_replacements_internal(
            anonymized_text
        )
        common_name_matches = []
        search_text_for_common = anonymized_text
        common_names_regex = self.regex_patterns["common_first_names"]
        for match in common_names_regex.finditer(search_text_for_common):
            name_found = match.group(0)
            if not any(
                name_found.lower() == org_k.lower()
                for org_k in self.org_chart_map_config
            ):
                if name_found.capitalize() in self.common_first_names_list:
                    common_name_matches.append(name_found)
        common_name_matches.sort(key=len, reverse=True)
        temp_common_name_map = {}
        for name_val in common_name_matches:
            if name_val not in self.anonymization_map:
                placeholder = self._generate_placeholder("person")
                self.anonymization_map[name_val] = placeholder
            temp_common_name_map[name_val] = self.anonymization_map[name_val]
        anonymized_text = self._apply_replacements_from_map(
            anonymized_text,
            temp_common_name_map,
            sorted(temp_common_name_map.keys(), key=len, reverse=True),
        )
        return anonymized_text

    def _anonymize_text_single_entry(self, text: str) -> str:
        """Anonymizes a single text entry (wrapper)."""
        return self._anonymize_text_single_entry_main_logic(text)

    def _manage_tone_single_entry_process_sentence(
        self, sentence_stripped: str
    ) -> tuple[bool, str]:
        """Checks a single sentence for tone issues."""
        for keyword_list, reason_template in [
            (self.tone_keywords_subjective, "Subjective_Opinion"),
            (self.tone_keywords_negative_emotion, "Negative_Emotion"),
            (self.tone_keywords_strong_doubt, "Strong_Doubt"),
            (self.tone_keywords_informal, "Informal_Language"),
        ]:
            for keyword in keyword_list:
                pattern = r"\b" + re.escape(keyword) + r"\b"
                if re.search(pattern, sentence_stripped, re.IGNORECASE):
                    return True, reason_template

        if self.regex_personal_opinion.search(sentence_stripped):
            exclusion = (
                r"\b(?:Je|J\')\s+(?:suis|ai|vois|dois|peux|vais|sais|fus"
                r"|serai|saurai)\b"
            )
            if not re.search(exclusion, sentence_stripped, re.IGNORECASE):
                return True, "Personal_Statement_Structure"
        return False, ""

    def _manage_tone_single_entry(self, text: str) -> tuple[list[dict[str, str]], bool]:
        """Manages tone for a single text entry, flagging issues."""
        tone_issues: list[dict[str, str]] = []
        requires_review = False
        sentences = re.split(r"(?<=[.!?])\s+", text) if text else []

        for sentence in sentences:
            sentence_stripped = sentence.strip()
            if not sentence_stripped:
                continue
            flagged_for_sentence, reason_for_sentence = (
                self._manage_tone_single_entry_process_sentence(sentence_stripped)
            )
            if flagged_for_sentence:
                issue_detail = {
                    "original_passage": sentence_stripped,
                    "flag_reason": reason_for_sentence,
                }
                tone_issues.append(issue_detail)
                requires_review = True
                log_passage = sentence_stripped[:50]
                logger.debug(
                    "Problème de ton '%s' marqué: '%.50s...'",
                    reason_for_sentence,
                    log_passage,
                )
        if requires_review:
            logger.info(
                "Entry requires tone review. Issues found: %d", len(tone_issues)
            )
        return tone_issues, requires_review

    def process_entries(
        self,
        raw_entry_dicts: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, str]]:
        """Processes all entries for anonymization and tone."""
        processed_entry_dicts: list[dict[str, Any]] = []
        self.counters = self._initialize_counters_from_map(self.anonymization_map)

        for entry_data_dict in raw_entry_dicts:
            source_file = entry_data_dict.get("source_file", "Inconnu")
            logger.debug("Sanitizing entry from '%s'", source_file)
            try:
                # Utiliser 'text' comme clé attendue par RawJournalEntryForTest
                current_raw_text = entry_data_dict["text"]
            except KeyError:
                logger.error("Champ 'text' manquant pour %s.", source_file)
                failed_entry_copy = entry_data_dict.copy()
                # Ligne 395 (E501) - Assurer que la chaîne est coupée si message long
                failed_entry_copy["anonymized_text"] = failed_entry_copy.get("text", "")
                failed_entry_copy["tone_issues"] = [
                    {
                        "original_passage": "MISSING_TEXT_FIELD",
                        "flag_reason": "Missing 'text' field",
                    }
                ]
                failed_entry_copy["requires_tone_review"] = True
                processed_entry_dicts.append(failed_entry_copy)
                continue
            except ValidationError as e:
                logger.error("Erreur validation Pydantic pour %s: %s", source_file, e)
                continue

            anonymized_text = self._anonymize_text_single_entry(current_raw_text)
            (tone_issues, requires_tone_review) = self._manage_tone_single_entry(
                anonymized_text
            )
            # Construire un dictionnaire à partir du modèle Pydantic
            # pour assurer la présence des champs de RawJournalEntryForTest
            # avant d'ajouter les nouveaux.
            try:
                # Ligne 423 (E501) - Coupée
                base_data_dict = RawJournalEntryForTest(
                    text=entry_data_dict["text"],
                    source_file=source_file,
                    entry_date=entry_data_dict.get("entry_date"),
                ).model_dump()
            except ValidationError as e:
                logger.error(
                    "Erreur création RawJournalEntryForTest pour %s: %s",
                    source_file,
                    e,  # Ligne 428 (E501) - message log coupé
                )
                # Ligne 429 & 430 (E501) - déjà OK
                continue  # Passer à l'entrée suivante

            output_dict = base_data_dict
            output_dict["anonymized_text"] = anonymized_text
            output_dict["tone_issues"] = tone_issues
            output_dict["requires_tone_review"] = requires_tone_review
            try:
                processed_entry_obj = ProcessedJournalEntryForTest(**output_dict)
                processed_entry_dicts.append(processed_entry_obj.model_dump())
            except ValidationError as e:
                logger.error(
                    "Erreur validation ProcessedJournalEntryForTest pour %s: %s",
                    source_file,
                    e,
                )
                processed_entry_dicts.append(output_dict)

        logger.info(
            "Anonymisation et gestion du ton terminées pour %d entrées.",
            len(processed_entry_dicts),
        )
        return processed_entry_dicts, self.anonymization_map


# ---- FIN DE LA CLASSE TextSanitizationExpert ----


@pytest.fixture()
def sanitizer_expert_fixture():
    """Provides a TextSanitizationExpert instance for testing."""  # D200: OK
    return TextSanitizationExpert(initial_anonymization_map={})


@pytest.fixture()
def sample_raw_entry_list() -> list[RawJournalEntryForTest]:
    """Provides a sample list of raw journal entries for testing."""
    entry1_text = (
        "Réunion avec Jérôme Carecchio et TP au sujet du Projet Alpha. "
        "Gecina est content. Arthur Perrot a dit que Magic Lemp aiderait. "
        "Brahim était là aussi. C'est un truc de fou mais je pense que ça va marcher."
    )
    entry2_text = (
        "Alexandre Morel a présenté le Projet Héraclès. C'était super. "
        "Jérôme était moins convaincu. Ce problème me frustre."
    )
    entry3_text = (
        "Discussion avec Yann et Nicolas sur un nouveau projet Projet Beta. "
        "Beñat Ortega a donné son accord. Expertime nous a contacté. "
        "Je suis pas sûr de tout ça."
    )
    return [
        RawJournalEntryForTest(
            text=entry1_text, source_file="test1.docx", entry_date=date(2023, 5, 15)
        ),
        RawJournalEntryForTest(
            text=entry2_text, source_file="test2.docx", entry_date=date(2023, 5, 16)
        ),
        RawJournalEntryForTest(
            text=entry3_text, source_file="test3.docx", entry_date=date(2023, 5, 17)
        ),
    ]


def test_initialization(sanitizer_expert_fixture: TextSanitizationExpert):
    """Tests basic initialization of the sanitizer."""
    assert sanitizer_expert_fixture is not None
    dsi_placeholder = "[Le_DSI]"
    assert dsi_placeholder in (sanitizer_expert_fixture.org_chart_map_config.values())
    assert "Magic Lemp" in sanitizer_expert_fixture.other_known_entities_config


# ... (le reste des tests comme avant)
# F841 pour counters et max_suffix_val dans _initialize_counters_from_map
# est corrigé en utilisant les variables.

# Les tests suivants utilisent la structure que vous avez copiée.
# Assurez-vous qu'ils appellent sanitizer_expert_fixture.process_entries
# en passant une liste de dictionnaires (obtenus via .model_dump())
# et en traitant la sortie (qui est une liste de dictionnaires et la map).
