# Interface Web Streamlit

Interface web simple pour l'OCR avec raisonnement.

## Lancement

```bash
# Option 1: Script de lancement
python app/run_app.py

# Option 2: Streamlit direct
streamlit run app/streamlit_app.py

# Option 3: Depuis le répertoire app
cd app
streamlit run streamlit_app.py
```

## Fonctionnalités

- 📄 Upload de documents (images, PDF)
- 🔍 OCR avec extraction de texte
- 🧠 Raisonnement avec prompts personnalisés
- 📊 Données structurées en JSON
- 📥 Téléchargement des résultats
- ⚙️ Configuration des modèles
- 📈 Monitoring mémoire

## Interface

- **Sidebar**: Configuration et gestion des modèles
- **Colonne gauche**: Upload et prompts
- **Colonne droite**: Résultats avec onglets

## Exemples de prompts

- "Extraire toutes les informations personnelles"
- "Extraire le montant, la date et le numéro de facture"
- "Extraire les compétences et l'expérience professionnelle"