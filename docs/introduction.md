# NEW-OCR Documentation

## Introduction

NEW-OCR est un système d'extraction de données intelligent qui combine OCR et IA pour extraire des informations structurées à partir d'images et de documents.

### Objectif Principal
- **OCR Simple** : Extraction de texte brut à partir d'images
- **Extraction IA** : Analyse intelligente et extraction de données spécifiques selon vos instructions

### Technologies Utilisées
- **GOT-OCR** : Modèle OCR avancé pour la reconnaissance de texte
- **Streamlit** : Interface web interactive
- **Modèles IA** : Extraction intelligente (Mistral, GPT, Qwen)

### Trois Modes d'Utilisation

#### 🌐 **Interface Web (Streamlit)**
- Interface utilisateur simple via navigateur
- Upload d'images par glisser-déposer
- Résultats en temps réel

#### 🖥️ **CLI (Command Line)**
- Utilisation en ligne de commande
- Idéal pour scripts et automatisation
- Traitement par lots

#### 🔌 **API REST**
- Intégration dans vos applications
- Endpoint `/process` unifié
- Format JSON standard

### Cas d'Usage Typiques
- Extraction de données de factures
- Analyse de documents administratifs
- Numérisation de formulaires
- Extraction d'informations produits

### Démarrage Rapide
```bash
# Installation
git clone https://github.com/ahmed00078/New-OCR.git
cd NEW-OCR
pip install -r requirements.txt

# Test CLI
python main.py process document.pdf

# Interface web
streamlit run app/streamlit_app.py
```