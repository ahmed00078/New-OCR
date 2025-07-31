#!/usr/bin/env python3
"""
Script de test pour l'extraction d'informations de rapports de soutenabilité
Développé pour tester la pipeline GOT-OCR sur des documents variés

Auteur: Equipe GOT-OCR
Date: 2025-07-31
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd

# Ajouter le répertoire racine au path
sys.path.insert(0, str(Path(__file__).parent))

from core.pipeline import pipeline

class SustainabilityReportExtractor:
    """Extracteur d'informations pour rapports de soutenabilité"""
    
    def __init__(self):
        self.data_dir = Path("data")
        self.results = []
        
        # Champs à extraire selon les spécifications
        self.target_fields = {
            "fabricant": "Nom du fabricant/constructeur",
            "annee": "Année du rapport ou du produit",
            "nom_produit": "Nom/modèle du produit",
            "impact_carbone": "Impact carbone (kg CO2 eq ou unité similaire)",
            "consommation_electrique": "Consommation électrique maximale (W, kW, etc.)",
            "poids_produit": "Poids du produit (kg, g, etc.)"
        }
    
    def get_extraction_prompt(self, strategy: str = "all_at_once") -> str:
        """
        Génère le prompt d'extraction selon la stratégie
        
        Args:
            strategy: "all_at_once" ou "step_by_step"
        """
        if strategy == "all_at_once":
            return """Extraire les informations suivantes du rapport de soutenabilité :
            
- Fabricant (nom de l'entreprise/constructeur)
- Année (du rapport ou du produit)
- Nom du produit (modèle exact)
- Impact carbone (en kg CO2 eq ou équivalent)
- Consommation électrique maximale (en W, kW ou équivalent)
- Poids du produit (en kg, g ou équivalent)

Retourner un JSON avec ces clés exactes : fabricant, annee, nom_produit, impact_carbone, consommation_electrique, poids_produit"""
        
        else:  # step_by_step
            return """Extraire uniquement le fabricant, l'année et le nom du produit de ce rapport de soutenabilité.
            
Retourner un JSON avec : fabricant, annee, nom_produit"""
    
    def process_single_document(self, pdf_path: Path, strategy: str = "all_at_once") -> Dict[str, Any]:
        """
        Traite un seul document PDF
        
        Args:
            pdf_path: Chemin vers le PDF
            strategy: Stratégie d'extraction
            
        Returns:
            Résultats d'extraction avec métadonnées
        """
        print(f"\n📄 Traitement: {pdf_path.name}")
        
        start_time = time.time()
        
        try:
            # Extraction avec la pipeline
            prompt = self.get_extraction_prompt(strategy)
            result = pipeline.process_simple(str(pdf_path), prompt)
            
            processing_time = time.time() - start_time
            
            # Préparation du résultat
            extraction_result = {
                "fichier": pdf_path.name,
                "strategie": strategy,
                "temps_traitement": round(processing_time, 2),
                "success": result.get("success", False),
                "donnees_extraites": result.get("structured_data", {}),
                "texte_brut_apercu": result.get("text", "")[:200] + "..." if result.get("text") else "",
                "pages_traitees": result.get("pages_processed", 0)
            }
            
            # Validation des données extraites
            if extraction_result["donnees_extraites"]:
                extraction_result["completude"] = self._calculate_completeness(
                    extraction_result["donnees_extraites"]
                )
            else:
                extraction_result["completude"] = 0.0
            
            print(f"✅ Succès - {processing_time:.1f}s - Complétude: {extraction_result.get('completude', 0):.1f}%")
            
            return extraction_result
            
        except Exception as e:
            processing_time = time.time() - start_time
            print(f"❌ Erreur: {e}")
            
            return {
                "fichier": pdf_path.name,
                "strategie": strategy,
                "temps_traitement": round(processing_time, 2),
                "success": False,
                "erreur": str(e),
                "donnees_extraites": {},
                "completude": 0.0
            }
    
    def _calculate_completeness(self, extracted_data: Dict[str, Any]) -> float:
        """Calcule le pourcentage de complétude des données extraites"""
        if not extracted_data:
            return 0.0
        
        expected_fields = set(self.target_fields.keys())
        extracted_fields = set(extracted_data.keys())
        
        # Compter les champs non-null et non-vides
        valid_fields = 0
        for field in expected_fields:
            value = extracted_data.get(field)
            if value and value != "null" and str(value).strip():
                valid_fields += 1
        
        return (valid_fields / len(expected_fields)) * 100
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """
        Lance un test complet sur tous les documents
        """
        print("🚀 Démarrage du test complet des rapports de soutenabilité")
        print("=" * 60)
        
        # Vérifier que le dossier existe
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Dossier data non trouvé: {self.data_dir}")
        
        # Lister les PDFs
        pdf_files = list(self.data_dir.glob("*.pdf"))
        if not pdf_files:
            raise FileNotFoundError("Aucun fichier PDF trouvé dans le dossier data")
        
        print(f"📁 {len(pdf_files)} fichiers PDF trouvés")
        
        # Test avec stratégie "tout d'un coup"
        print("\n🎯 Test 1: Extraction complète (tout d'un coup)")
        for pdf_file in pdf_files:
            result = self.process_single_document(pdf_file, "all_at_once")
            self.results.append(result)
        
        # Analyse des résultats
        analysis = self._analyze_results()
        
        return {
            "nombre_documents": len(pdf_files),
            "resultats_detailles": self.results,
            "analyse_globale": analysis
        }
    
    def _analyze_results(self) -> Dict[str, Any]:
        """Analyse les résultats globaux"""
        if not self.results:
            return {}
        
        total_docs = len(self.results)
        successful_docs = sum(1 for r in self.results if r["success"])
        
        completeness_scores = [r.get("completude", 0) for r in self.results if r["success"]]
        avg_completeness = sum(completeness_scores) / len(completeness_scores) if completeness_scores else 0
        
        processing_times = [r["temps_traitement"] for r in self.results]
        avg_processing_time = sum(processing_times) / len(processing_times)
        
        return {
            "taux_succes": (successful_docs / total_docs) * 100,
            "completude_moyenne": round(avg_completeness, 1),
            "temps_traitement_moyen": round(avg_processing_time, 1),
            "temps_total": round(sum(processing_times), 1),
            "documents_reussis": successful_docs,
            "documents_total": total_docs
        }
    
    def save_results(self, output_file: str = "sustainability_extraction_results.json"):
        """Sauvegarde les résultats en JSON"""
        results_data = {
            "meta": {
                "script": "test_sustainability_reports.py",
                "date_execution": time.strftime("%Y-%m-%d %H:%M:%S"),
                "pipeline_version": "1.0.0"
            },
            "configuration": {
                "champs_cibles": self.target_fields,
                "dossier_source": str(self.data_dir)
            },
            "resultats": self.results,
            "analyse": self._analyze_results()
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Résultats sauvegardés: {output_file}")
    
    def generate_report(self):
        """Génère un rapport de synthèse"""
        analysis = self._analyze_results()
        
        print(f"\n📊 RAPPORT DE SYNTHÈSE")
        print("=" * 40)
        print(f"Documents traités: {analysis['documents_total']}")
        print(f"Taux de succès: {analysis['taux_succes']:.1f}%")
        print(f"Complétude moyenne: {analysis['completude_moyenne']:.1f}%")
        print(f"Temps moyen par document: {analysis['temps_traitement_moyen']:.1f}s")
        print(f"Temps total: {analysis['temps_total']:.1f}s")
        
        print(f"\n🎯 PERFORMANCE PAR CHAMP:")
        # Analyser la performance par champ
        field_stats = {}
        for field in self.target_fields.keys():
            found_count = sum(1 for r in self.results 
                            if r["success"] and r["donnees_extraites"].get(field))
            field_stats[field] = (found_count / len([r for r in self.results if r["success"]])) * 100 if self.results else 0
        
        for field, percentage in field_stats.items():
            print(f"  {field}: {percentage:.1f}%")

def main():
    """Fonction principale"""
    extractor = SustainabilityReportExtractor()
    
    try:
        # Lancer le test complet
        results = extractor.run_comprehensive_test()
        
        # Générer le rapport
        extractor.generate_report()
        
        # Sauvegarder les résultats
        extractor.save_results()
        
        print(f"\n🎉 Test terminé avec succès!")
        
    except Exception as e:
        print(f"\n❌ Erreur lors du test: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()