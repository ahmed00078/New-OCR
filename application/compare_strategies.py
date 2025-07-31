#!/usr/bin/env python3
"""
Comparaison des stratégies d'extraction:
1. Tout d'un coup (all-at-once)
2. Étape par étape (step-by-step)

Comme demandé par l'encadrant: "voir si ton pipeline arrive a bien extraire les valeurs d'un coup 
ou si cela marche mieux en plusieurs fois"
"""

import sys
import json
import time
from pathlib import Path

# Ajouter le répertoire racine au path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.pipeline import pipeline

def strategy_all_at_once(pdf_path: str) -> dict:
    """Stratégie 1: Extraction complète en une fois"""
    
    prompt = """Extraire TOUTES les informations suivantes du rapport de soutenabilité :

- Fabricant (nom de l'entreprise/constructeur)
- Année (du rapport ou du produit)
- Nom du produit (modèle exact)
- Impact carbone (en kg CO2 eq ou équivalent)
- Consommation électrique maximale (en W, kW ou équivalent)
- Poids du produit (en kg, g ou équivalent)

Retourner un JSON avec ces clés exactes : fabricant, annee, nom_produit, impact_carbone, consommation_electrique, poids_produit"""
    
    start_time = time.time()
    result = pipeline.process_simple(pdf_path, prompt)
    processing_time = time.time() - start_time
    
    return {
        "strategie": "tout_d_un_coup",
        "temps": processing_time,
        "success": result.get("success", False),
        "donnees": result.get("structured_data", {}),
        "pages": result.get("pages_processed", 0)
    }

def strategy_step_by_step(pdf_path: str) -> dict:
    """Stratégie 2: Extraction étape par étape"""
    
    start_time = time.time()
    
    # Étape 1: Informations de base
    prompt1 = """Extraire uniquement les informations de base du rapport :
- Fabricant (nom de l'entreprise/constructeur)
- Année (du rapport ou du produit)
- Nom du produit (modèle exact)

Retourner un JSON avec : fabricant, annee, nom_produit"""
    
    result1 = pipeline.process_simple(pdf_path, prompt1)
    
    # Étape 2: Métriques environnementales
    prompt2 = """Extraire uniquement les métriques environnementales et techniques :
- Impact carbone (en kg CO2 eq ou équivalent)
- Consommation électrique maximale (en W, kW ou équivalent)  
- Poids du produit (en kg, g ou équivalent)

Retourner un JSON avec : impact_carbone, consommation_electrique, poids_produit"""
    
    result2 = pipeline.process_simple(pdf_path, prompt2)
    
    processing_time = time.time() - start_time
    
    # Combiner les résultats
    combined_data = {}
    if result1.get("structured_data"):
        combined_data.update(result1["structured_data"])
    if result2.get("structured_data"):
        combined_data.update(result2["structured_data"])
    
    return {
        "strategie": "etape_par_etape",
        "temps": processing_time,
        "success": result1.get("success", False) and result2.get("success", False),
        "donnees": combined_data,
        "pages": result1.get("pages_processed", 0),
        "etapes": {
            "etape1": result1.get("structured_data", {}),
            "etape2": result2.get("structured_data", {})
        }
    }

def compare_strategies(filename: str):
    """Compare les deux stratégies sur un fichier"""
    
    data_dir = Path("data")
    pdf_path = data_dir / filename
    
    if not pdf_path.exists():
        print(f"❌ Fichier non trouvé: {pdf_path}")
        return
    
    print(f"⚖️  COMPARAISON DES STRATÉGIES")
    print(f"📄 Fichier: {filename}")
    print("=" * 50)
    
    # Test stratégie 1
    print("🎯 Stratégie 1: Tout d'un coup...")
    result1 = strategy_all_at_once(str(pdf_path))
    
    # Test stratégie 2  
    print("🔄 Stratégie 2: Étape par étape...")
    result2 = strategy_step_by_step(str(pdf_path))
    
    # Comparaison
    print("\n📊 RÉSULTATS COMPARATIFS")
    print("-" * 30)
    
    print(f"⏱️  TEMPS:")
    print(f"  Tout d'un coup: {result1['temps']:.1f}s")
    print(f"  Étape par étape: {result2['temps']:.1f}s")
    print(f"  Différence: {result2['temps'] - result1['temps']:+.1f}s")
    
    print(f"\n✅ SUCCÈS:")
    print(f"  Tout d'un coup: {'Oui' if result1['success'] else 'Non'}")
    print(f"  Étape par étape: {'Oui' if result2['success'] else 'Non'}")
    
    # Calculer complétude
    expected_fields = ["fabricant", "annee", "nom_produit", "impact_carbone", "consommation_electrique", "poids_produit"]
    
    def calculate_completeness(data):
        if not data:
            return 0
        valid_fields = sum(1 for field in expected_fields 
                          if data.get(field) and str(data[field]).strip() and data[field] != "null")
        return (valid_fields / len(expected_fields)) * 100
    
    comp1 = calculate_completeness(result1['donnees'])
    comp2 = calculate_completeness(result2['donnees'])
    
    print(f"\n📈 COMPLÉTUDE:")
    print(f"  Tout d'un coup: {comp1:.1f}%")
    print(f"  Étape par étape: {comp2:.1f}%")
    print(f"  Différence: {comp2 - comp1:+.1f}%")
    
    print(f"\n📋 DONNÉES EXTRAITES:")
    print("  Tout d'un coup:")
    if result1['donnees']:
        for field, value in result1['donnees'].items():
            print(f"    {field}: {value}")
    else:
        print("    Aucune donnée")
    
    print("  Étape par étape:")
    if result2['donnees']:
        for field, value in result2['donnees'].items():
            print(f"    {field}: {value}")
    else:
        print("    Aucune donnée")
    
    # Recommandation
    print(f"\n🎯 RECOMMANDATION:")
    if comp1 > comp2:
        print("  → Stratégie 'Tout d'un coup' plus efficace")
    elif comp2 > comp1:
        print("  → Stratégie 'Étape par étape' plus efficace")
    else:
        if result1['temps'] < result2['temps']:
            print("  → Stratégie 'Tout d'un coup' plus rapide")
        else:
            print("  → Performances équivalentes")
    
    # Sauvegarder
    comparison_result = {
        "fichier": filename,
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "strategies": {
            "tout_d_un_coup": result1,
            "etape_par_etape": result2
        },
        "analyse": {
            "completude_s1": comp1,
            "completude_s2": comp2,
            "temps_s1": result1['temps'],
            "temps_s2": result2['temps'],
            "meilleure_strategie": "tout_d_un_coup" if comp1 >= comp2 else "etape_par_etape"
        }
    }
    
    output_file = f"comparison_{filename.replace('.pdf', '')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(comparison_result, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Résultats sauvegardés: {output_file}")

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
            choice = int(input("\nChoisir un fichier pour comparaison (numéro): ")) - 1
            filename = pdf_files[choice].name
        except (ValueError, IndexError):
            print("❌ Choix invalide")
            return
    
    compare_strategies(filename)

if __name__ == "__main__":
    main()