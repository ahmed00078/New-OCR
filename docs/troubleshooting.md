# Troubleshooting

## Erreurs d'installation

### ❌ `No module named 'core'`
```bash
# Solution : Lancer depuis le répertoire racine
cd /path/to/NEW-OCR
python main.py process document.pdf
```

### ❌ `CUDA out of memory`
```python
# config/settings.py
MAX_MEMORY_MB = 4000  # Réduire
DEVICE = "cpu"        # Forcer CPU
```

### ❌ `torch not found`
```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### ❌ `Reasoning failed`
**Causes :**
- Prompt trop complexe
- Texte OCR vide
- Modèle IA surchargé

**Solutions :**
```bash
# Test sans raisonnement
python main.py process document.pdf  # Sans --prompt

# Prompt plus simple
python main.py process document.pdf --prompt "extraire nom"
```

### 🔥 **GPU surchauffe**
```bash
# Surveiller température
nvidia-smi

# Forcer CPU temporairement
export CUDA_VISIBLE_DEVICES=""
python main.py process document.pdf
```

## Interface Web

### ❌ `Streamlit connection error`
```bash
# Vérifier port disponible
lsof -i :8501

# Changer port
streamlit run app/streamlit_app.py --server.port 8502
```

### ❌ `Upload failed`
**Causes :**
- Fichier trop volumineux
- Format non supporté

**Solutions :**
- Réduire taille image
- Convertir en format supporté (PDF, PNG, JPG)

## API Issues

### ❌ `Connection refused`
```bash
# Vérifier serveur API lancé
curl http://localhost:8000/health

# Relancer serveur
python api/server.py
```

### ❌ `Request timeout`
```python
# Augmenter timeout
# api/server.py
timeout = 300  # 5 minutes
```

## Diagnostic rapide

### Script de diagnostic
```bash
python scripts/diagnose.py
```

### Vérifications manuelles
```bash
# Python version
python --version

# Modules installés
pip list | grep torch
pip list | grep streamlit

# Espace disque
df -h

# Mémoire RAM
free -h

# GPU (si NVIDIA)
nvidia-smi
```