# NEW-OCR

Système d'extraction de données intelligent qui combine OCR et IA pour extraire des informations structurées à partir d'images et documents.

## 🎯 Objectifs

- **OCR Avancé** : Extraction de texte avec GOT-OCR 2.0
- **Extraction IA** : Analyse intelligente et extraction de données spécifiques
- **Interface Multiple** : CLI, Web (Streamlit), et API REST
- **Gestion Optimisée** : Mémoire intelligente et performance GPU/CPU
- **Focus Environnemental** : Spécialisé pour impact carbone et données techniques

## 🏗️ Architecture

```
NEW-OCR/
├── core/
│   ├── models/
│   │   ├── ocr_model.py          # GOT-OCR wrapper
│   │   └── reasoning_engine.py   # IA pour extraction structurée
│   └── pipeline.py               # Pipeline principale
├── app/
│   └── streamlit_app.py          # Interface web Streamlit
├── api/
│   ├── server.py                 # API REST
│   └── models.py                 # Modèles Pydantic
├── config/
│   └── settings.py               # Configuration centralisée
├── docs/                         # Documentation complète
└── main.py                       # CLI principal
```

## 🚀 Installation

```bash
# Installation rapide
git clone https://github.com/ahmed00078/New-OCR.git
cd NEW-OCR
python3.10 -m venv venv310
source venv310/bin/activate
pip install -r requirements.txt

# Test
python test_pipeline.py
```

### Configuration GPU (optionnel)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 📖 Usage

### 🖥️ CLI (Command Line)
```bash
# OCR simple
python main.py process document.pdf

# Extraction IA avec prompt
python main.py process document.pdf --prompt "extraire l'impact carbone et le poids du produit"

# Avec options avancées
python main.py process document.pdf \
  --prompt "extraire empreinte carbone, consommation énergie" \
  --use-layout --format json --output resultat.json
```

### 🌐 Interface Web (Streamlit)
```bash
# Lancer interface web
streamlit run app/streamlit_app.py

# Accès : http://localhost:8501
```

### 🔌 API REST
```bash
# Démarrer serveur API
python api/server.py

# Test
curl -X POST "http://localhost:8000/process" \
  -F "file=@document.pdf" \
  -F "prompt=extraire impact carbone"
```

## ⚙️ Configuration

Fichier principal : `config/settings.py`
```python
# Modèles
OCR_MODEL = "got-ocr2_0"
REASONING_MODEL = "mistral-7b"  # ou qwen2.5-7b

# Performance  
DEVICE = "auto"  # auto, cuda, cpu
MAX_MEMORY_MB = 16000

# Spécialisations
FOCUS = "environmental"  # impact carbone, données techniques
```

## 📊 Exemples

### 1. Document technique d'équipement
```bash
python main.py process equipement.pdf \
  --prompt "extraire nom fabricant, impact carbone, poids produit" \
  --format json
```

**Résultat :**
```json
{
  "nom_fabricant": "Fabricant XYZ",
  "impact_carbone": "150 kg CO2",
  "poids_produit": "1.5 kg"
}
```

### 2. Rapport environnemental
```bash
python main.py process rapport.pdf \
  --prompt "extraire empreinte carbone, consommation énergie"
```

### 3. Interface Web
1. Glisser-déposer document
2. Prompt : `"extraire impact environnemental"`
3. Télécharger résultats JSON

## 📚 Documentation

Documentation complète disponible dans `/docs/` :
- **[Introduction](docs/introduction.md)** - Vue d'ensemble
- **[Installation](docs/installation.md)** - Guide détaillé  
- **[Utilisation](docs/utilisation.md)** - CLI, Web, API
- **[Configuration](docs/configuration.md)** - Paramétrage
- **[Exemples](docs/exemples.md)** - Cas d'usage détaillés
- **[Troubleshooting](docs/troubleshooting.md)** - Résolution problèmes
- **[Architecture](docs/architecture.md)** - Technique

## ⚡ Optimisations

### Gestion Mémoire
- **Lazy loading** : Modèles chargés à la demande
- **Auto unload** : Déchargement automatique
- **Memory monitoring** : Surveillance RAM en temps réel

### Performance
- **GPU/CPU auto** : Détection automatique device optimal
- **Batch processing** : Traitement optimisé gros documents
- **Cache intelligent** : Mise en cache résultats

## 🐛 Support

**Problèmes courants :**
- Mémoire insuffisante → Réduire `MAX_MEMORY_MB`
- Erreur CUDA → Forcer `DEVICE=cpu`
- Import failed → Lancer depuis répertoire racine

**Support complet :** [docs/troubleshooting.md](docs/troubleshooting.md)

## 🚀 Fonctionnalités

✅ OCR multi-format (PDF, images)  
✅ Extraction IA avec prompts  
✅ Interface CLI complète  
✅ Interface web Streamlit  
✅ API REST  
✅ Gestion mémoire optimisée  
✅ Support GPU/CPU  
✅ Focus données environnementales  

## 🤝 Crédits

Basé sur :
- [GOT-OCR 2.0](https://github.com/Ucas-HaoranWei/GOT-OCR2.0) - OCR
- [Mistral](https://mistral.ai/) / [Qwen](https://github.com/QwenLM/Qwen2.5) - IA
- [Streamlit](https://streamlit.io/) - Interface web