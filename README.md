# CV Parsing — ATS Intelligent

Deux approches complémentaires pour le parsing de CV.

---

## 1. Parser Classique (`cv_parser_classic.py`)
**Regex + SpaCy NER**

| Composant | Rôle |
|---|---|
| `TextExtractor` | Lit PDF / DOCX / TXT |
| `RegexPatterns` | Email, téléphone, LinkedIn, GitHub, dates |
| `NERExtractor` | Nom, localisation, organisations (SpaCy) |
| `SectionSplitter` | Découpe le CV en sections logiques |
| `SkillsExtractor` | Matching sur dictionnaire de 50+ skills |
| `ExperienceParser` | Parse les blocs d'expérience heuristiquement |
| `EducationParser` | Parse les blocs de formation |

```bash
pip install spacy pdfminer.six python-docx
python -m spacy download fr_core_news_lg
python cv_parser_classic.py mon_cv.pdf
```

**Avantages** : rapide, 100% local, déterministe, pas de GPU  
**Limites** : fragile face aux CV non structurés, miss les infos sémantiques

---

## 2. Parser LLM (`cv_parser_llm.py`)
**Ollama (Mistral / LLaMA3 / Gemma2)**

| Composant | Rôle |
|---|---|
| `CVPromptBuilder` | Construit le prompt avec JSON Schema intégré |
| `OllamaClient` | Appel local au LLM via Ollama |
| `JSONResponseParser` | Extrait et valide le JSON retourné |
| `LLMCVParser` | Pipeline : LLM + fallback Regex |
| `CVParserFactory` | Auto-sélection du meilleur modèle dispo |

```bash
pip install ollama pdfminer.six python-docx
# Installer Ollama : https://ollama.com
ollama pull mistral
python cv_parser_llm.py mon_cv.pdf mistral
```

**Avantages** : comprend les CV non structurés, extrait le contexte sémantique  
**Limites** : plus lent, nécessite Ollama installé

---

## Sortie commune (`CVData`)

Les deux parsers retournent la même structure `CVData` :

```json
{
  "full_name": "Marie Dupont",
  "email": "marie@example.com",
  "phone": "+33 6 12 34 56 78",
  "location": "Paris",
  "linkedin": "linkedin.com/in/marie-dupont",
  "github": "github.com/mariedupont",
  "summary": "Ingénieure Data avec 5 ans d'expérience...",
  "skills": ["python", "pytorch", "docker", "spark"],
  "languages": [{"language": "Français", "level": "Natif"}],
  "experiences": [
    {
      "title": "Data Scientist",
      "company": "Acme Corp",
      "period": "Jan 2022 - Présent",
      "description": "Développement de modèles ML..."
    }
  ],
  "education": [
    {
      "degree": "Master Data Science",
      "institution": "Université Paris-Saclay",
      "period": "2019 - 2021"
    }
  ],
  "certifications": ["AWS Certified ML Specialty"]
}
```

---

## Architecture ATS recommandée

```
CV (PDF/DOCX)
     │
     ▼
TextExtractor
     │
     ├──► ClassicCVParser  (contacts, skills → haute précision)
     │
     └──► LLMCVParser      (expériences, résumé → haute compréhension)
     │
     ▼
CVData (fusionné)
     │
     ▼
ATS Database / Matching Engine
```
