#!/usr/bin/env python3
"""
Test rapide pour un document spécifique
Usage: python quick_test.py [nom_fichier.pdf]
"""

import sys
import json
from pathlib import Path

# Ajouter le répertoire racine au path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.pipeline import pipeline

def test_single_file(filename: str):
    """Test rapide d'un seul fichier"""
    
    data_dir = Path("data")
    pdf_path = data_dir / filename
    
    if not pdf_path.exists():
        print(f"❌ Fichier non trouvé: {pdf_path}")
        available_files = list(data_dir.glob("*.pdf"))
        print(f"📁 Fichiers disponibles:")
        for f in available_files:
            print(f"  - {f.name}")
        return
    
    print(f"🔍 Test du fichier: {filename}")
    print("-" * 40)
    
    # Prompt d'extraction
    prompt = """Extraire les informations suivantes du rapport de soutenabilité :

- Fabricant (nom de l'entreprise/constructeur)
- Année (du rapport ou du produit)  
- Nom du produit (modèle exact)
- Impact carbone (en kg CO2 eq ou équivalent)
- Consommation électrique maximale (en W, kW ou équivalent)
- Poids du produit (en kg, g ou équivalent)

Retourner un JSON avec ces clés exactes : fabricant, annee, nom_produit, impact_carbone, consommation_electrique, poids_produit"""
    
    try:
        # Traitement
        result = pipeline.process_simple(str(pdf_path), prompt)
        
        # Affichage des résultats
        print(f"✅ Succès: {result.get('success', False)}")
        print(f"📄 Pages traitées: {result.get('pages_processed', 0)}")
        
        # AFFICHAGE COMPLET DU TEXTE EXTRAIT
        if result.get("text"):
            print(f"\n📝 TEXTE COMPLET EXTRAIT PAR GOT-OCR:")
            print("=" * 60)
            print(result["text"])
            print("=" * 60)
        else:
            print(f"\n⚠️  Aucun texte extrait par GOT-OCR")
        
        # AFFICHAGE DE LA RÉPONSE BRUTE DU MODÈLE DE RAISONNEMENT
        if result.get("structured_data") and "raw_response" in result["structured_data"]:
            print(f"\n🤖 RÉPONSE BRUTE DU MODÈLE DE RAISONNEMENT:")
            print("-" * 60)
            print(result["structured_data"]["raw_response"])
            print("-" * 60)
        
        # Données extraites finales
        if result.get("structured_data"):
            print(f"\n📊 DONNÉES STRUCTURÉES FINALES:")
            print(json.dumps(result["structured_data"], indent=2, ensure_ascii=False))
        else:
            print(f"\n⚠️  Aucune donnée structurée extraite")
        
    except Exception as e:
        print(f"❌ Erreur: {e}")

def main():
    """Fonction principale"""
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        # Liste les fichiers disponibles
        data_dir = Path("data")
        pdf_files = list(data_dir.glob("*.pdf"))
        
        print("📁 Fichiers PDF disponibles:")
        for i, f in enumerate(pdf_files, 1):
            print(f"  {i}. {f.name}")
        
        try:
            choice = int(input("\nChoisir un fichier (numéro): ")) - 1
            filename = pdf_files[choice].name
        except (ValueError, IndexError):
            print("❌ Choix invalide")
            return
    
    test_single_file(filename)

if __name__ == "__main__":
    main()