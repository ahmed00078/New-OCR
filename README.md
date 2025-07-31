# GOT-OCR Simplified Pipeline

Version simplifiée et corrigée d'une pipeline OCR basée sur **GOT-OCR 2.0** avec support de raisonnement et segmentation d'images.

## 🎯 Objectifs

- **Pipeline OCR complète** : Extraction de texte avec GOT-OCR 2.0
- **Segmentation intelligente** : PP-DocLayout pour identifier les zones (texte, tableau, image, formule)
- **Raisonnement contextuel** : AI pour transformer le texte en JSON structuré
- **Gestion mémoire optimisée** : Traitement par batch, cache intelligent
- **API simple** : Interface REST minimaliste

## 🏗️ Architecture

```
got-ocr-simplified/
├── core/
│   ├── models/
│   │   ├── ocr_engine.py         # GOT-OCR wrapper
│   │   ├── layout_engine.py      # PP-DocLayout wrapper
│   │   └── reasoning_engine.py   # AI wrapper
│   ├── processors/
│   │   ├── document_processor.py # Orchestrateur principal
│   │   ├── pdf_handler.py        # Gestion PDF optimisée
│   │   └── renderer.py           # Rendu HTML/LaTeX
│   └── pipeline.py               # API publique simple
├── api/
│   ├── server.py                 # FastAPI minimal
│   └── models.py                 # Pydantic models
├── config/
│   └── settings.py               # Configuration centralisée
└── main.py                       # Point d'entrée CLI
```

## 🚀 Installation

### 1. Prérequis
```bash
python >= 3.8
CUDA (optionnel, pour GPU)
```

### 2. Installation des dépendances
```bash
# Cloner le repo
cd NEW-OCR

# Créer environnement virtuel
python -m venv venv
source venv/bin/activate  # ou venv\\Scripts\\activate sur Windows

# Installer les dépendances
pip install -r requirements.txt
```

### 3. Configuration
Les modèles sont configurés dans `config/settings.py` :
- **GOT-OCR** : `stepfun-ai/GOT-OCR-2.0-hf`
- **Layout** : `PP-DocLayout_plus-L` 
- **Reasoning** : `Qwen/Qwen2.5-7B-Instruct`

## 📖 Usage

### Interface CLI

```bash
# Serveur API
python main.py server --host 0.0.0.0 --port 8000

# Traitement de fichier
python main.py process document.pdf --prompt "extraire nom, âge, adresse"

# OCR simple
python main.py process image.jpg --format markdown

# Informations système
python main.py info
```

### API REST

```bash
# Démarrer le serveur
python main.py server

# Test avec curl
curl -X POST "http://localhost:8000/process-simple" \
  -F "file=@document.pdf" \
  -F "user_prompt=extraire les noms et montants"
```

### Usage programmatique

```python
from core.pipeline import pipeline

# OCR simple
result = pipeline.process_ocr_only("document.pdf")
print(result)

# Extraction avec raisonnement
result = pipeline.process_simple(
    "facture.pdf", 
    "extraire numéro facture, montant, date, client"
)
print(result["structured_data"])
```

## 🔧 Configuration

Variables d'environnement disponibles :
```bash
OCR_MODEL=stepfun-ai/GOT-OCR-2.0-hf
LAYOUT_MODEL=PP-DocLayout_plus-L
REASONING_MODEL=Qwen/Qwen2.5-7B-Instruct
MAX_MEMORY_MB=20000
BATCH_SIZE=5
MAX_PAGES=50
DEVICE=auto
```

## 📊 Exemples

### 1. Extraction de données de facture
```python
result = pipeline.process_simple(
    "facture.pdf",
    "extraire: numéro facture, date, client, montant total, TVA"
)

# Résultat JSON structuré :
{
  "numero_facture": "INV-2024-001",
  "date": "2024-01-15",
  "client": "Société ABC",
  "montant_total": "1234.56",
  "tva": "206.09"
}
```

### 2. Analyse de CV
```python
result = pipeline.process_simple(
    "cv.pdf",
    "extraire: nom, email, téléphone, expériences, compétences"
)
```

### 3. OCR avec format spécialisé
```python
# Pour documents scientifiques avec formules
text = pipeline.process_ocr_only("article.pdf", format_type="latex")

# Pour documentation technique  
text = pipeline.process_ocr_only("manual.pdf", format_type="markdown")
```

## 🧪 Tests

```bash
# Lancer les tests
python test_pipeline.py

# Test de l'API
curl http://localhost:8000/health
```

## 🔧 Optimisations

### Gestion mémoire
- **Lazy loading** : Modèles chargés à la demande
- **Cache intelligent** : Un seul modèle en mémoire à la fois
- **Traitement par batch** : Évite les OOM sur gros PDF
- **Cleanup automatique** : Nettoyage mémoire après traitement

### Performance
- **Segmentation adaptative** : Layout detection uniquement si nécessaire
- **Troncature intelligente** : Respect des limites de contexte LLM
- **Fallbacks robustes** : Alternatives si modèles principaux indisponibles

## 🐛 Résolution de problèmes

### Erreurs communes
1. **Mémoire insuffisante** : Réduire `BATCH_SIZE` ou `MAX_MEMORY_MB`
2. **Modèles non trouvés** : Vérifier connexion internet et cache HuggingFace
3. **PDF non supporté** : Installer `pdf2image` et `poppler`
4. **GPU non détecté** : Forcer `DEVICE=cpu`

### Logs
```bash
# Mode verbose
python main.py --verbose process document.pdf

# Logs API
python main.py server --reload
```

## 📈 Roadmap

- [ ] Support multi-langues
- [ ] Optimisations quantization
- [ ] Batch processing API
- [ ] Interface web simple
- [ ] Métriques de performance

## 🤝 Contribution

Pipeline développée pour usage interne. Basée sur :
- [GOT-OCR 2.0](https://github.com/Ucas-HaoranWei/GOT-OCR2.0)
- [PP-DocLayout](https://github.com/PaddlePaddle/PaddleDetection)
- [Qwen](https://github.com/QwenLM/Qwen2.5)

## 📄 License

Projet éducatif - Usage interne uniquement