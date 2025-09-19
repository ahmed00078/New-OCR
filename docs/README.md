# NEW-OCR Documentation

Documentation complète pour NEW-OCR - Système d'extraction de données intelligent.

## Navigation rapide

### 📚 **Documentation principale**
- **[Introduction](introduction.md)** - Vue d'ensemble et démarrage rapide
- **[Installation](installation.md)** - Guide d'installation étape par étape  
- **[Utilisation](utilisation.md)** - CLI, Interface Web et API
- **[Configuration](configuration.md)** - Paramétrage et optimisation
- **[Exemples](exemples.md)** - Cas d'usage concrets avec commandes
- **[Troubleshooting](troubleshooting.md)** - Résolution de problèmes

### 🏗️ **Technique**
- **[Architecture](architecture.md)** - Composants et flux de données

## Démarrage ultra-rapide

```bash
# 1. Installation
git clone https://github.com/ahmed00078/New-OCR.git
cd NEW-OCR
pip install -r requirements.txt

# 2. Test CLI
python main.py process document.pdf --prompt "extraire nom, email"

# 3. Interface web  
streamlit run app/streamlit_app.py
```

## Structure projet

```
NEW-OCR/
├── docs/           # Documentation complète
├── core/           # Pipeline principale
├── config/         # Configuration
├── app/            # Interface Streamlit
├── api/            # API REST
├── models/         # Cache modèles IA
└── examples/       # Exemples documents
```

## Support

- **Issues** : [GitHub Issues](https://github.com/ahmed00078/New-OCR/issues)
- **Documentation** : `/docs/`
- **Exemples** : `/docs/exemples.md`

---

*Documentation mise à jour régulièrement - Version actuelle compatible avec NEW-OCR v1.0*