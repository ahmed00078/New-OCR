# Configuration

## Fichier principal : `config/settings.py`

### Configuration de base
```python
class Settings:
    # Modèles
    OCR_MODEL = "got-ocr2_0"
    REASONING_MODEL = "mistral-7b"
    
    # Performance
    DEVICE = "auto"  # auto, cuda, cpu
    MAX_MEMORY_MB = 16000
    
    # Chemins
    MODEL_CACHE_DIR = "./models"
    TEMP_DIR = "./temp"
```

## Modèles disponibles

### OCR Models
- `got-ocr2_0` : Modèle principal (recommandé)

### Reasoning Models
- `mistral-7b` : Équilibre performance/qualité
- `qwen2.5-7b` : Alternative open-source

## Configuration GPU

### Automatique (recommandé)
```python
DEVICE = "auto"
```

### Manuel
```python
# GPU uniquement
DEVICE = "cuda"

# CPU uniquement  
DEVICE = "cpu"

# GPU spécifique
DEVICE = "cuda:0"
```