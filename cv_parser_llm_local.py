"""
CV Parser LLM — Ollama (Open Source)
======================================
Approche : extraction structurée via un LLM open source (Mistral, LLaMA3, etc.)
tournant en local avec Ollama.

Dépendances :
    pip install ollama pdfminer.six python-docx
    # Installer Ollama : https://ollama.com
    # Puis : ollama pull mistral  (ou llama3, gemma2, etc.)
"""

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Any

import ollama
from pdfminer.high_level import extract_text as pdf_extract_text
from docx import Document


# ---------------------------------------------------------------------------
# Data Model (identique à la version classique pour compatibilité ATS)
# ---------------------------------------------------------------------------

@dataclass
class CVData:
    full_name:      Optional[str]  = None
    email:          Optional[str]  = None
    phone:          Optional[str]  = None
    location:       Optional[str]  = None
    linkedin:       Optional[str]  = None
    github:         Optional[str]  = None
    summary:        Optional[str]  = None
    skills:         list[str]      = field(default_factory=list)
    languages:      list[str]      = field(default_factory=list)
    experiences:    list[dict]     = field(default_factory=list)
    education:      list[dict]     = field(default_factory=list)
    certifications: list[str]      = field(default_factory=list)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=indent)

    @classmethod
    def from_dict(cls, data: dict) -> "CVData":
        return cls(
            full_name      = data.get("full_name"),
            email          = data.get("email"),
            phone          = data.get("phone"),
            location       = data.get("location"),
            linkedin       = data.get("linkedin"),
            github         = data.get("github"),
            summary        = data.get("summary"),
            skills         = data.get("skills", []),
            languages      = data.get("languages", []),
            experiences    = data.get("experiences", []),
            education      = data.get("education", []),
            certifications = data.get("certifications", []),
        )


# ---------------------------------------------------------------------------
# Text Extractor (réutilisable)
# ---------------------------------------------------------------------------

class TextExtractor:
    """Extrait le texte brut depuis PDF, DOCX ou TXT."""

    @staticmethod
    def extract(file_path: str) -> str:
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            return pdf_extract_text(file_path) or ""
        elif suffix == ".docx":
            doc = Document(file_path)
            return "\n".join(p.text for p in doc.paragraphs)
        elif suffix in (".txt", ".md"):
            return path.read_text(encoding="utf-8")
        else:
            raise ValueError(f"Format non supporté : {suffix}")


# ---------------------------------------------------------------------------
# Prompt Builder
# ---------------------------------------------------------------------------

class CVPromptBuilder:
    """
    Construit un prompt structuré pour l'extraction du CV.
    On utilise le JSON Schema directement dans le prompt pour guider
    la réponse du LLM (structured output prompting).
    """

    # Schéma JSON cible explicite dans le prompt
    JSON_SCHEMA = """
{
  "full_name":      "string | null",
  "email":          "string | null",
  "phone":          "string | null",
  "location":       "string | null",
  "linkedin":       "string | null",
  "github":         "string | null",
  "summary":        "string | null  (2-3 phrases max)",
  "skills":         ["string", "..."],
  "languages":      [{"language": "string", "level": "string"}, "..."],
  "experiences": [
    {
      "title":       "string",
      "company":     "string",
      "location":    "string | null",
      "period":      "string  (ex: Jan 2022 - Présent)",
      "description": "string  (missions clés, bullet points)"
    }
  ],
  "education": [
    {
      "degree":      "string",
      "institution": "string",
      "location":    "string | null",
      "period":      "string"
    }
  ],
  "certifications": ["string", "..."]
}
"""

    def build(self, cv_text: str) -> str:
        # On tronque le CV si trop long (évite de dépasser la context window)
        truncated = cv_text[:6000] if len(cv_text) > 6000 else cv_text

        return f"""Tu es un expert en analyse de CV et recrutement.

Analyse le CV ci-dessous et extrais TOUTES les informations en respectant EXACTEMENT le schéma JSON fourni.

RÈGLES IMPORTANTES :
- Réponds UNIQUEMENT avec du JSON valide, sans markdown, sans explication.
- Si une information est absente, utilise `null` pour les champs scalaires et `[]` pour les tableaux.
- Ne complète pas, n'invente pas d'informations manquantes.
- Normalise les dates au format "MMM YYYY" si possible.
- Pour les compétences, liste uniquement les compétences techniques explicitement mentionnées.

SCHÉMA JSON ATTENDU :
{self.JSON_SCHEMA}

CV À ANALYSER :
---
{truncated}
---

JSON :"""


# ---------------------------------------------------------------------------
# JSON Response Parser
# ---------------------------------------------------------------------------

class JSONResponseParser:
    """Extrait et valide le JSON de la réponse LLM."""

    @staticmethod
    def parse(raw_response: str) -> dict[str, Any]:
        # Nettoie le markdown si le LLM l'a quand même ajouté
        cleaned = re.sub(r"```(?:json)?", "", raw_response).replace("```", "").strip()

        # Tente d'extraire le JSON depuis la première accolade
        json_start = cleaned.find("{")
        json_end   = cleaned.rfind("}") + 1

        if json_start == -1 or json_end == 0:
            raise ValueError(f"Aucun JSON trouvé dans la réponse : {raw_response[:200]}")

        json_str = cleaned[json_start:json_end]

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON invalide : {e}\nContenu : {json_str[:300]}")


# ---------------------------------------------------------------------------
# Ollama LLM Client
# ---------------------------------------------------------------------------

class OllamaClient:
    """
    Wrapper Ollama pour l'inférence locale.
    Modèles recommandés par ordre de performance/vitesse :
        - mistral        (7B, rapide, bon équilibre)
        - llama3         (8B, excellent pour l'extraction structurée)
        - gemma2         (9B, très bon en JSON)
        - mixtral        (47B MoE, meilleur mais plus lent)
    """

    def __init__(self, model: str = "mistral", temperature: float = 0.0):
        self.model       = model
        self.temperature = temperature  # 0.0 = déterministe, idéal pour extraction

    def generate(self, prompt: str) -> str:
        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": self.temperature},
        )
        return response["message"]["content"]

    def is_available(self) -> bool:
        """Vérifie que le modèle est disponible localement."""
        try:
            models = ollama.list()
            available = [m["name"].split(":")[0] for m in models.get("models", [])]
            return self.model in available
        except Exception:
            return False


# ---------------------------------------------------------------------------
# LLM CV Parser — Pipeline Principal
# ---------------------------------------------------------------------------

class LLMCVParser:
    """
    Parser CV basé sur LLM (Ollama).
    
    Stratégie en deux passes :
    1. Extraction principale par LLM (informations sémantiques complexes)
    2. Post-processing Regex pour garantir email/phone/URL (haute précision)
    """

    # Patterns de fallback pour les champs critiques
    _EMAIL_PATTERN   = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
    _PHONE_PATTERN   = re.compile(r"(?:\+?\d{1,3}[\s\-.]?)?(?:\(?\d{2,4}\)?[\s\-.]?){2,4}\d{2,4}")
    _LINKEDIN_PATTERN = re.compile(r"linkedin\.com/in/[\w\-]+")
    _GITHUB_PATTERN  = re.compile(r"github\.com/[\w\-]+")

    def __init__(self, model: str = "mistral", temperature: float = 0.0):
        self.llm_client    = OllamaClient(model=model, temperature=temperature)
        self.prompt_builder = CVPromptBuilder()
        self.json_parser   = JSONResponseParser()

    def parse(self, file_path: str) -> CVData:
        # 1. Extraction du texte brut
        raw_text = TextExtractor.extract(file_path)

        # 2. Construction du prompt et appel LLM
        prompt      = self.prompt_builder.build(raw_text)
        raw_response = self.llm_client.generate(prompt)

        # 3. Parsing de la réponse JSON
        extracted_data = self.json_parser.parse(raw_response)

        # 4. Post-processing : fallback Regex pour les champs critiques
        extracted_data = self._apply_regex_fallback(extracted_data, raw_text)

        # 5. Validation et nettoyage
        extracted_data = self._clean_data(extracted_data)

        return CVData.from_dict(extracted_data)

    def _apply_regex_fallback(self, data: dict, raw_text: str) -> dict:
        """
        Si le LLM rate un champ critique (email, phone, etc.),
        on utilise Regex comme filet de sécurité.
        """
        if not data.get("email"):
            m = self._EMAIL_PATTERN.search(raw_text)
            data["email"] = m.group(0) if m else None

        if not data.get("phone"):
            m = self._PHONE_PATTERN.search(raw_text)
            data["phone"] = m.group(0).strip() if m else None

        if not data.get("linkedin"):
            m = self._LINKEDIN_PATTERN.search(raw_text)
            data["linkedin"] = m.group(0) if m else None

        if not data.get("github"):
            m = self._GITHUB_PATTERN.search(raw_text)
            data["github"] = m.group(0) if m else None

        return data

    def _clean_data(self, data: dict) -> dict:
        """Normalise et nettoie les données extraites."""
        # Assure que les listes sont bien des listes
        for list_field in ["skills", "languages", "experiences", "education", "certifications"]:
            if not isinstance(data.get(list_field), list):
                data[list_field] = []

        # Nettoie les strings vides
        for str_field in ["full_name", "email", "phone", "location", "linkedin", "github", "summary"]:
            val = data.get(str_field)
            if isinstance(val, str) and not val.strip():
                data[str_field] = None

        # Déduplique les skills
        if data.get("skills"):
            data["skills"] = sorted(set(s.lower().strip() for s in data["skills"] if s))

        return data


# ---------------------------------------------------------------------------
# Factory : choisir le bon modèle selon les ressources disponibles
# ---------------------------------------------------------------------------

class CVParserFactory:
    """
    Sélectionne automatiquement le meilleur modèle disponible.
    Ordre de préférence : llama3 > mistral > gemma2 > phi3
    """

    MODELS_BY_PRIORITY = ["llama3", "mistral", "gemma2", "phi3"]

    @classmethod
    def create(cls, preferred_model: Optional[str] = None) -> LLMCVParser:
        if preferred_model:
            return LLMCVParser(model=preferred_model)

        for model in cls.MODELS_BY_PRIORITY:
            client = OllamaClient(model=model)
            if client.is_available():
                print(f"[CVParserFactory] Modèle sélectionné : {model}")
                return LLMCVParser(model=model)

        # Fallback sur mistral même s'il n'est pas listé (pull automatique possible)
        print("[CVParserFactory] Aucun modèle détecté, tentative avec mistral...")
        return LLMCVParser(model="mistral")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python cv_parser_llm.py <chemin_du_cv> [modele_ollama]")
        print("Exemple: python cv_parser_llm.py mon_cv.pdf mistral")
        sys.exit(1)

    file_path     = sys.argv[1]
    model_name    = sys.argv[2] if len(sys.argv) > 2 else None

    parser = CVParserFactory.create(preferred_model=model_name)
    result = parser.parse(file_path)
    print(result.to_json())