# Architecture

## Vue d'ensemble

NEW-OCR suit une architecture modulaire simple avec 3 composants principaux :

```
📄 Input → 🔍 OCR → 🧠 AI → 📊 Output
```

## Composants

### Core Pipeline (`core/pipeline.py`)
Orchestrateur principal qui coordonne tous les modules.

### OCR Module (`core/models/ocr_model.py`)
- Modèle GOT-OCR pour extraction de texte
- Support images et PDF
- Analyse layout optionnelle

### Reasoning Engine (`core/models/reasoning_engine.py`)
- IA pour extraction structurée
- Support multi-modèles (Mistral, GPT, Qwen)
- Traitement via prompts utilisateur

## Flux de données

1. **Input** : Upload document (CLI/Web/API)
2. **OCR** : Extraction texte brut + layout
3. **Reasoning** : Extraction données structurées (optionnel)
4. **Output** : Formatage résultats (JSON/text/markdown)

## Interfaces

### CLI (`main.py`)
```bash
python main.py process document.pdf --prompt "extraire l'impact carbone et le poids du produit"
```

### Web App (`app/streamlit_app.py`)
Interface Streamlit avec upload et visualisation

### API (`api/`)
Endpoints REST pour intégration externe

## Gestion Ressources

- **Lazy Loading** : Modèles chargés à la demande
- **Memory Monitor** : Surveillance utilisation RAM
- **Auto Unload** : Déchargement automatique
- **GPU/CPU** : Détection automatique device optimal