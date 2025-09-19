# Installation

## Prérequis
- Python 3.10+
- 16GB RAM minimum

## Installation Rapide

```bash
# 1. Cloner repository
git clone https://github.com/ahmed00078/New-OCR.git
cd NEW-OCR

# 2. Environnement virtuel
python3.10 -m venv venv310
source venv310/bin/activate

# 3. Dependencies
pip install -r requirements.txt
```

## Configuration GPU (Optionnel)

```bash
# NVIDIA CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Vérifier
python -c "import torch; print(torch.cuda.is_available())"
```

## Docker

```bash
# Build
docker build -t new-ocr .

# Run
docker run -p 8501:8501 new-ocr
```

## Vérification

```bash
# Test complet
python test_pipeline.py

# Test CLI
python main.py process --help

# Test web
streamlit run app/streamlit_app.py
```

## Problèmes Courants

### Erreur CUDA
```bash
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Mémoire insuffisante
```python
# config/settings.py
MAX_MEMORY_MB = 4000  # Réduire
DEVICE = "cpu"        # Forcer CPU
```