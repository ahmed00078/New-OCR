# Exemples

## 1. Facture

### Document
Image de document technique d'un equipement électronique.

### Commande CLI
```bash
python main.py process document_technique.pdf \
  --prompt "Extraire nom du fabricant, l'impact carbone et le poids du produit" \
  --format json
```

### Résultat
```json
{
  "nom_fabricant": "Fabricant XYZ",
  "impact_carbone": "150 kg CO2",
  "poids_produit": "1.5 kg"
}
```

## 2. CV / Curriculum Vitae

### Commande
```bash
python main.py process cv.pdf \
  --prompt "Extraire nom, email, téléphone, compétences, expérience" \
  --use-layout
```

### Résultat
```json
{
  "nom": "Marie Martin",
  "email": "marie.martin@email.com",
  "telephone": "+33 6 12 34 56 78",
  "competences": ["Python", "Machine Learning", "SQL"],
  "experience": [
    {
      "poste": "Data Scientist",
      "entreprise": "TechCorp",
      "periode": "2022-2024"
    }
  ]
}
```

## 3. Document d'identité

### Commande
```bash
python main.py process carte_id.jpg \
  --prompt "Extraire nom, prénom, date naissance, lieu naissance" \
  --format json
```

### Résultat
```json
{
  "nom": "DUBOIS",
  "prenom": "Pierre",
  "date_naissance": "12/03/1985",
  "lieu_naissance": "Paris"
}
```

## 4. Formulaire administratif

### Commande
```bash
python main.py process formulaire.pdf \
  --prompt "Extraire toutes les informations remplies dans le formulaire"
```

### Résultat
```json
{
  "nom_famille": "LAURENT",
  "prenom": "Sophie",
  "adresse": "123 Rue de la République, 75001 Paris",
  "profession": "Ingénieur",
  "situation_familiale": "Mariée",
  "nombre_enfants": "2"
}
```

## 5. Traitement par lots

### Commande
```bash
# Traiter tous les PDF d'un dossier
for file in documents/*.pdf; do
  python main.py process "$file" \
    --prompt "Extraire informations importantes" \
    --output "resultats/$(basename "$file" .pdf).json"
done
```

## Interface Web - Exemples

### Upload et prompt
1. Glisser-déposer le document
2. Saisir prompt : `"Extraire nom, email, téléphone"`
3. Activer "Layout analysis" si document structuré
4. Cliquer "Traiter"

### Prompts suggérés
- Documents administratifs : `"Extraire toutes les informations personnelles"`
- Factures : `"Extraire montant, date, fournisseur, client"`
- CVs : `"Extraire compétences, expérience, formation"`
- Formulaires : `"Extraire tous les champs remplis"`