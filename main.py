#!/usr/bin/env python3
"""
GOT-OCR Simplified Pipeline
Version simplifiee d'une pipeline OCR basee sur GOT-OCR 2.0
avec support de raisonnement et segmentation d'images.
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# Ajouter le repertoire racine au path
sys.path.insert(0, str(Path(__file__).parent))

from core.pipeline import pipeline
from config.settings import settings

def setup_logging(level=logging.INFO):
    """Configure le logging"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    """Point d'entree principal"""
    parser = argparse.ArgumentParser(
        description="GOT-OCR Pipeline - OCR avec raisonnement et segmentation"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commandes disponibles')
    
    # Commande server
    server_parser = subparsers.add_parser('server', help='Lancer le serveur API')
    server_parser.add_argument('--host', default=settings.HOST, help='Host du serveur')
    server_parser.add_argument('--port', type=int, default=settings.PORT, help='Port du serveur')
    server_parser.add_argument('--reload', action='store_true', help='Mode developpement avec rechargement auto')
    
    # Commande process
    process_parser = subparsers.add_parser('process', help='Traiter un fichier')
    process_parser.add_argument('file', help='Chemin vers le fichier à traiter')
    process_parser.add_argument('--prompt', help='Instructions pour extraction d\'informations')
    process_parser.add_argument('--no-layout', action='store_true', help='Desactiver la segmentation de layout')
    process_parser.add_argument('--format', choices=['plain', 'markdown', 'latex'], 
                               default='plain', help='Format de sortie OCR')
    process_parser.add_argument('--output', choices=['json', 'html', 'markdown'], 
                               default='json', help='Format de la reponse')
    process_parser.add_argument('--save', help='Sauvegarder le resultat dans un fichier')
    
    # Commande info
    info_parser = subparsers.add_parser('info', help='Informations systeme')
    
    # Options globales
    parser.add_argument('--verbose', '-v', action='store_true', help='Mode verbeux')
    parser.add_argument('--quiet', '-q', action='store_true', help='Mode silencieux')
    
    args = parser.parse_args()
    
    # Configuration du logging
    if args.quiet:
        setup_logging(logging.WARNING)
    elif args.verbose:
        setup_logging(logging.DEBUG)
    else:
        setup_logging(logging.INFO)
    
    logger = logging.getLogger(__name__)
    
    if args.command == 'server':
        # Lancer le serveur
        import uvicorn
        from api.server import app
        
        logger.info(f"Starting server on {args.host}:{args.port}")
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level="info" if not args.quiet else "warning"
        )
    
    elif args.command == 'process':
        # Traiter un fichier
        if not os.path.exists(args.file):
            logger.error(f"File not found: {args.file}")
            sys.exit(1)
        
        logger.info(f"Processing file: {args.file}")
        
        try:
            result = pipeline.process(
                input_source=args.file,
                user_prompt=args.prompt,
                use_layout=not args.no_layout,
                format_type=args.format,
                output_format=args.output
            )
            
            if args.save:
                # Sauvegarder dans un fichier
                import json
                with open(args.save, 'w', encoding='utf-8') as f:
                    if args.output == 'json':
                        json.dump(result, f, ensure_ascii=False, indent=2)
                    elif args.output == 'html':
                        f.write(result.get('html', ''))
                    elif args.output == 'markdown':
                        f.write(result.get('markdown', ''))
                
                logger.info(f"Result saved to: {args.save}")
            else:
                # Afficher le resultat
                if args.output == 'json':
                    import json
                    print(json.dumps(result, ensure_ascii=False, indent=2))
                elif args.output == 'html':
                    print(result.get('html', ''))
                elif args.output == 'markdown':
                    print(result.get('markdown', ''))
        
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            sys.exit(1)
    
    elif args.command == 'info':
        # Afficher les informations systeme
        print("=== GOT-OCR Pipeline Info ===")
        print(f"Version: 1.0.0")
        print(f"Configuration:")
        print(f"  OCR Model: {settings.OCR_MODEL}")
        print(f"  Layout Model: {settings.LAYOUT_MODEL}")
        print(f"  Reasoning Model: {settings.REASONING_MODEL}")
        print(f"  Max Memory: {settings.MAX_MEMORY_MB} MB")
        print(f"  Batch Size: {settings.BATCH_SIZE}")
        print(f"  Max Pages: {settings.MAX_PAGES}")
        print(f"  Device: {settings.DEVICE}")
        
        # Informations memoire
        try:
            memory_info = pipeline.get_memory_info()
            print(f"Memory Usage: {memory_info['current_memory_mb']:.1f} MB")
            print(f"Memory Limit: {memory_info['max_memory_mb']} MB")
            print(f"Memory Usage: {memory_info['memory_usage_percent']:.1f}%")
        except Exception as e:
            print(f"Could not get memory info: {e}")
    
    else:
        # Afficher l'aide si aucune commande
        parser.print_help()

if __name__ == "__main__":
    main()