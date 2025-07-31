#!/usr/bin/env python3
"""
Script de test pour valider la pipeline GOT-OCR
"""

import sys
import os
import tempfile
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Ajouter le répertoire racine au path
sys.path.insert(0, str(Path(__file__).parent))

def create_test_image():
    """Crée une image de test avec du texte"""
    # Créer une image blanche avec du texte
    img = Image.new('RGB', (800, 600), color='white')
    draw = ImageDraw.Draw(img)
    
    # Essayer d'utiliser une police, sinon police par défaut
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    # Ajouter du texte de test
    test_text = """Document de Test OCR
    
Nom: Jean Dupont
Âge: 35 ans
Adresse: 123 Rue de la Paix, Paris
Email: jean.dupont@email.com
Téléphone: +33 1 23 45 67 89

Montant: 1,234.56 €
Date: 15/12/2024

Ce document contient des informations
structurées pour tester l'extraction."""
    
    # Dessiner le texte
    y_position = 50
    for line in test_text.split('\n'):
        draw.text((50, y_position), line.strip(), fill='black', font=font)
        y_position += 30
    
    return img

def test_basic_ocr():
    """Test OCR basique"""
    print("=== Test OCR Basique ===")
    
    try:
        from core.pipeline import pipeline
        
        # Créer image de test
        test_img = create_test_image()
        
        # Test OCR simple
        result = pipeline.process(
            input_source=test_img,
            user_prompt=None,
            use_layout=False,
            format_type="plain",
            output_format="json"
        )
        
        print(f"Succès: {result['success']}")
        print(f"Texte extrait: {result['text'][:100]}...")
        print(f"Pages traitées: {result['pages_processed']}")
        
        return result['success']
        
    except Exception as e:
        print(f"Erreur: {e}")
        return False

def test_extraction_with_reasoning():
    """Test extraction avec raisonnement"""
    print("\n=== Test Extraction avec Raisonnement ===")
    
    try:
        from core.pipeline import pipeline
        
        # Créer image de test
        test_img = create_test_image()
        
        # Test avec extraction d'informations
        user_prompt = "Extraire le nom, l'âge, l'email, le téléphone et le montant"
        
        result = pipeline.process(
            input_source=test_img,
            user_prompt=user_prompt,
            use_layout=True,
            format_type="plain",
            output_format="json"
        )
        
        print(f"Succès: {result['success']}")
        print(f"Données structurées: {json.dumps(result.get('structured_data', {}), indent=2, ensure_ascii=False)}")
        
        return result['success'] and result.get('structured_data') is not None
        
    except Exception as e:
        print(f"Erreur: {e}")
        return False

def test_memory_management():
    """Test gestion mémoire"""
    print("\n=== Test Gestion Mémoire ===")
    
    try:
        from core.pipeline import pipeline
        
        # Informations mémoire
        memory_info = pipeline.get_memory_info()
        print(f"Mémoire actuelle: {memory_info['current_memory_mb']:.1f} MB")
        print(f"Limite mémoire: {memory_info['max_memory_mb']} MB")
        print(f"Usage: {memory_info['memory_usage_percent']:.1f}%")
        
        # Test déchargement des modèles
        pipeline.unload_models()
        print("Modèles déchargés avec succès")
        
        return True
        
    except Exception as e:
        print(f"Erreur: {e}")
        return False

def test_api_import():
    """Test import des composants API"""
    print("\n=== Test Import API ===")
    
    try:
        from api.server import app
        from api.models import ProcessRequest, ProcessResponse
        print("Imports API réussis")
        return True
        
    except Exception as e:
        print(f"Erreur import API: {e}")
        return False

def test_configuration():
    """Test configuration"""
    print("\n=== Test Configuration ===")
    
    try:
        from config.settings import settings
        
        print(f"OCR Model: {settings.OCR_MODEL}")
        print(f"Layout Model: {settings.LAYOUT_MODEL}")
        print(f"Reasoning Model: {settings.REASONING_MODEL}")
        print(f"Max Memory: {settings.MAX_MEMORY_MB} MB")
        print(f"Device: {settings.DEVICE}")
        
        return True
        
    except Exception as e:
        print(f"Erreur configuration: {e}")
        return False

def main():
    """Fonction principale de test"""
    print("🚀 Test de la Pipeline GOT-OCR Simplifiée")
    print("=" * 50)
    
    tests = [
        ("Configuration", test_configuration),
        ("Import API", test_api_import),
        ("OCR Basique", test_basic_ocr),
        ("Extraction avec Raisonnement", test_extraction_with_reasoning),
        ("Gestion Mémoire", test_memory_management),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{status}")
        except Exception as e:
            print(f"❌ FAIL - Exception: {e}")
            results.append((test_name, False))
    
    # Résumé
    print("\n" + "=" * 50)
    print("📊 Résumé des Tests")
    print("=" * 50)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅" if success else "❌"
        print(f"{status} {test_name}")
    
    print(f"\n🎯 Résultat: {passed}/{total} tests passés")
    
    if passed == total:
        print("🎉 Tous les tests sont passés ! La pipeline est prête.")
    else:
        print("⚠️  Certains tests ont échoué. Vérifiez les dépendances et la configuration.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)