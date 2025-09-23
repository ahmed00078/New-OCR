# Script de Traitement en Lot - Pierre

Script simple pour traiter plusieurs fichiers avec l'API OCR en une seule fois.

## 🚀 Utilisation Simple

### Commande de base
```bash
python batch_processor.py DOSSIER_FICHIERS "VOTRE_PROMPT"
```

### Exemples concrets

**Rapports environnementaux :**
```bash
python batch_processor.py /docs/sustainability "Extract carbon footprint, power consumption, manufacturer, product weight"
```

## 📋 Paramètres

| Paramètre | Description | Obligatoire |
|-----------|-------------|-------------|
| `DOSSIER` | Dossier contenant les fichiers à traiter | ✅ |
| `PROMPT` | Instructions d'extraction | ✅ |
| `-o, --output` | Nom du fichier JSON de sortie | ❌ |
| `--api-url` | URL de l'API (défaut: localhost:8000) | ❌ |

## 📁 Fichiers Supportés

- PDF (`.pdf`)
- Images: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp`

## 📊 Fichier de Sortie

Le script génère un fichier JSON avec :

```json
{
  "meta": {
    "script": "batch_processor.py",
    "date_execution": "2025-09-23T14:21:52.943878",
    "user_prompt": "Extract screen size, carbon impact, energy consumption, recycling rate",
    "source_folder": "test/",
    "total_files": 4,
    "successful_files": 4,
    "failed_files": 0,
    "total_processing_time": 80.11
  },
  "results": [
    {
      "file": "sustainability_report_acer.pdf",
      "success": true,
      "data": {
        "carbon_footprint": 723,
        "power_consumption": 18.38,
        "manufacturer": "Acer",
        "product_weight": 10.6
      },
      "pages_processed": 1,
      "processing_time": 12.5,
      "error": null
    }
  ]
}
```

## ⚡ Conseils d'Utilisation

1. **Démarrer l'API d'abord :**
   ```bash
   python -m uvicorn api.server:app --host 0.0.0.0 --port 8000
   ```

2. **Tester avec un petit échantillon** avant de traiter beaucoup de fichiers

3. **Utiliser des prompts clairs** comme :
   - "Extract carbon footprint, power consumption, manufacturer"
   - "Extract screen size, energy efficiency, CO2 emissions"
   - "Extract environmental certifications, recycling rate, product lifetime"

## 🛠️ Dépannage

- **API non accessible** : Vérifier que le serveur API est démarré
- **Aucun fichier trouvé** : Vérifier le chemin du dossier
- **Erreurs de traitement** : Consulter les logs dans le fichier de sortie