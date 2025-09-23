import os
import json
import time
import requests
from pathlib import Path
import argparse
from datetime import datetime
from typing import List, Dict, Any

class BatchProcessor:
    """Processeur en lot pour l'API OCR"""

    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.results = []

    def process_file(self, file_path: str, user_prompt: str) -> Dict[str, Any]:
        """Traite un seul fichier"""
        print(f"📄 Traitement: {os.path.basename(file_path)}")

        try:
            with open(file_path, 'rb') as f:
                files = {'file': f}
                data = {
                    'mode': 'extract',
                    'user_prompt': user_prompt,
                    'use_layout': 'true',
                    'format_type': 'plain',
                    'output_format': 'json'
                }

                start_time = time.time()
                response = requests.post(f"{self.api_url}/process", files=files, data=data)
                processing_time = time.time() - start_time

                if response.status_code == 200:
                    result = response.json()
                    result['file_name'] = os.path.basename(file_path)
                    result['local_processing_time'] = processing_time
                    print(f"   ✅ Succès en {processing_time:.2f}s")
                    return result
                else:
                    error_result = {
                        'success': False,
                        'file_name': os.path.basename(file_path),
                        'error': f"HTTP {response.status_code}: {response.text}",
                        'local_processing_time': processing_time
                    }
                    print(f"   ❌ Erreur HTTP {response.status_code}")
                    return error_result

        except Exception as e:
            error_result = {
                'success': False,
                'file_name': os.path.basename(file_path),
                'error': str(e),
                'local_processing_time': 0
            }
            print(f"   ❌ Erreur: {e}")
            return error_result

    def get_supported_files(self, folder_path: str) -> List[str]:
        """Récupère tous les fichiers supportés dans le dossier"""
        supported_extensions = ['.pdf', '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        files = []

        for ext in supported_extensions:
            files.extend(Path(folder_path).glob(f"*{ext}"))
            files.extend(Path(folder_path).glob(f"*{ext.upper()}"))

        return [str(f) for f in sorted(files)]

    def process_batch(self, folder_path: str, user_prompt: str, output_file: str = None):
        """Traite tous les fichiers d'un dossier"""
        print(f"🚀 Démarrage du traitement en lot")
        print(f"📁 Dossier: {folder_path}")
        print(f"💬 Prompt: {user_prompt}")
        print("=" * 60)

        # Vérifier que l'API est accessible
        try:
            health_response = requests.get(f"{self.api_url}/health")
            if health_response.status_code != 200:
                print(f"❌ API non accessible: {self.api_url}")
                return
            print(f"✅ API accessible: {self.api_url}")
        except Exception as e:
            print(f"❌ Impossible de contacter l'API: {e}")
            return

        # Récupérer les fichiers
        files = self.get_supported_files(folder_path)
        if not files:
            print(f"❌ Aucun fichier supporté trouvé dans {folder_path}")
            return

        print(f"📊 {len(files)} fichiers trouvés")
        print()

        # Traiter chaque fichier
        start_time = time.time()

        for i, file_path in enumerate(files, 1):
            print(f"[{i}/{len(files)}] ", end="")
            result = self.process_file(file_path, user_prompt)
            self.results.append(result)
            print()

        total_time = time.time() - start_time

        # Statistiques
        successful = sum(1 for r in self.results if r.get('success', False))
        failed = len(self.results) - successful

        print("=" * 60)
        print(f"📈 Résultats:")
        print(f"   ✅ Réussis: {successful}/{len(files)}")
        print(f"   ❌ Échecs: {failed}/{len(files)}")
        print(f"   ⏱️  Temps total: {total_time:.2f}s")
        print(f"   📊 Temps moyen: {total_time/len(files):.2f}s par fichier")

        # Sauvegarder les résultats
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"resultats_batch_{timestamp}.json"

        self.save_results(output_file, user_prompt, folder_path, total_time)
        print(f"💾 Résultats sauvegardés: {output_file}")

    def save_results(self, output_file: str, user_prompt: str, folder_path: str, total_time: float):
        """Sauvegarde les résultats dans un fichier JSON"""
        # Formatter les résultats selon le format souhaité
        formatted_results = []

        for r in self.results:
            formatted_result = {
                "success": r.get('success', False),
                "file": r.get('file_name', ''),
                "text": r.get('text', ''),
                "data": r.get('structured_data'),
                "pages_processed": r.get('pages_processed', 0),
                "processing_time": r.get('processing_time', 0),
                "error": r.get('error')
            }
            formatted_results.append(formatted_result)

        output_data = {
            "meta": {
                "script": "batch_processor.py",
                "date_execution": datetime.now().isoformat(),
                "user_prompt": user_prompt,
                "source_folder": folder_path,
                "total_files": len(self.results),
                "successful_files": sum(1 for r in self.results if r.get('success', False)),
                "failed_files": sum(1 for r in self.results if not r.get('success', False)),
                "total_processing_time": total_time
            },
            "results": formatted_results
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(description="Traitement en lot de documents via l'API OCR")

    parser.add_argument(
        "folder",
        help="Dossier contenant les fichiers à traiter"
    )

    parser.add_argument(
        "prompt",
        help="Prompt d'extraction (ex: 'Extract screen size, carbon impact')"
    )

    parser.add_argument(
        "-o", "--output",
        help="Nom du fichier de sortie JSON (optionnel)",
        default=None
    )

    parser.add_argument(
        "--api-url",
        help="URL de l'API OCR",
        default="http://localhost:8000"
    )

    args = parser.parse_args()

    # Vérifier que le dossier existe
    if not os.path.exists(args.folder):
        print(f"❌ Dossier non trouvé: {args.folder}")
        return

    # Lancer le traitement
    processor = BatchProcessor(api_url=args.api_url)
    processor.process_batch(args.folder, args.prompt, args.output)

if __name__ == "__main__":
    main()