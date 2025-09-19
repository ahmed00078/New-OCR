# Utilisation

## CLI (Command Line)

### Commandes de base
```bash
# OCR simple
python main.py process document.pdf

# Avec extraction IA
python main.py process document.pdf --prompt "extraire l'impact carbone et le poids du produit"

# Options avancées
python main.py process document.pdf \
  --prompt "extraire l'impact carbone et le poids du produit" \
  --use-layout \
  --format json \
  --output resultat.json
```

### Options disponibles
- `--prompt` : Instructions d'extraction
- `--use-layout` : Analyse structure document
- `--format` : Format sortie (plain, json, markdown)
- `--output` : Fichier de sortie

## Interface Web

```bash
# Lancer interface
streamlit run app/streamlit_app.py
```

### Fonctionnalités
1. **Upload** : Glisser-déposer fichiers
2. **Prompt** : Saisir instructions extraction
3. **Options** : Layout, format, modèles
4. **Résultats** : Texte + données structurées
5. **Export** : Télécharger JSON/text

## API REST

### Démarrage serveur
```bash
python api/server.py
```

### Endpoints

#### POST `/process`
```bash
curl -X POST "http://localhost:8000/process" \
  -F "file=@document.pdf" \
  -F "prompt=extraire nom et email"
```

#### Réponse
```json
{
  "success": true,
  "text": "texte extrait...",
  "structured_data": {
    "nom": "Jean Dupont",
    "email": "jean@email.com"
  }
}
```

## Prompts Efficaces

### Structure
```
"Extraire [DONNÉES] de ce [TYPE_DOCUMENT]"
```

### Exemples
```bash
# Facture
--prompt "Extraire montant total, date, numéro facture"

# Impact environnemental
--prompt "Extraire empreinte carbone, consommation énergie"

# Document administratif
--prompt "Extraire nom, adresse, téléphone, email"
```

## Formats de sortie

### Plain Text
```bash
--format plain
```
Texte brut extrait par OCR

### JSON
```bash
--format json
```
Données structurées en JSON

### Markdown
```bash
--format markdown
```
Texte formaté avec mise en page préservée