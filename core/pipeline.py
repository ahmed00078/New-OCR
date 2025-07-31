import gc
import psutil
import torch
from typing import Optional, Dict, Any, Union
from PIL import Image
import logging

from core.processors.document_processor import DocumentProcessor, ProcessingResult
from core.processors.renderer import Renderer
from config.settings import settings

logger = logging.getLogger(__name__)

class MemoryManager:
    """Gestionnaire de memoire intelligent"""
    
    def __init__(self, max_memory_mb: int = None):
        self.max_memory_mb = max_memory_mb or settings.MAX_MEMORY_MB
        self.initial_memory = self.get_memory_usage()
        
    def get_memory_usage(self) -> float:
        """Retourne l'usage memoire actuel en MB"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0
    
    def check_memory_limit(self) -> bool:
        """Verifie si on approche de la limite memoire"""
        current = self.get_memory_usage()
        return current > self.max_memory_mb * 0.8  # 80% de la limite
    
    def cleanup_memory(self):
        """Force le nettoyage memoire"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info(f"Memory after cleanup: {self.get_memory_usage():.1f} MB")

class ModelCache:
    """Cache intelligent pour les modeles - un seul modele charge e la fois"""
    
    def __init__(self):
        self._current_model = None
        self._current_type = None
        self._models = {}
    
    def get_model(self, model_type: str, model_instance):
        """Retourne le modele demande, en dechargeant l'ancien si necessaire"""
        if self._current_type == model_type:
            return model_instance
        
        # Decharger le modele actuel
        if self._current_model is not None:
            logger.info(f"Unloading {self._current_type} model")
            self._current_model.unload()
        
        # Charger le nouveau modele
        logger.info(f"Loading {model_type} model")
        model_instance.load()
        
        self._current_model = model_instance
        self._current_type = model_type
        
        return model_instance
    
    def unload_all(self):
        """Decharge tous les modeles"""
        if self._current_model is not None:
            self._current_model.unload()
            self._current_model = None
            self._current_type = None

class OCRPipeline:
    """Pipeline principal simplifie pour GOT-OCR avec raisonnement"""
    
    def __init__(self):
        self.processor = DocumentProcessor()
        self.memory_manager = MemoryManager()
        self.model_cache = ModelCache()
        self.renderer = Renderer()
        
    def process(self,
                input_source: Union[str, Image.Image],
                user_prompt: Optional[str] = None,
                use_layout: bool = True,
                format_type: str = "plain",
                output_format: str = "json") -> Dict[str, Any]:
        """
        Traite un document ou une image
        
        Args:
            input_source: Chemin fichier ou image PIL
            user_prompt: Instructions pour extraction d'infos
            use_layout: Utiliser segmentation de layout  
            format_type: Format OCR ("plain", "markdown", "latex")
            output_format: Format de sortie ("json", "html", "markdown")
            
        Returns:
            Resultat formate selon output_format
        """
        try:
            logger.info("Starting OCR pipeline processing")
            
            # Verifier la memoire au debut
            if self.memory_manager.check_memory_limit():
                logger.warning("Memory usage high, performing cleanup")
                self.memory_manager.cleanup_memory()
            
            # Traitement selon le type d'entree
            if isinstance(input_source, str):
                # Fichier
                result = self.processor.process_document(
                    file_path=input_source,
                    user_prompt=user_prompt,
                    use_layout=use_layout,
                    format_type=format_type
                )
            else:
                # Image PIL
                result = self.processor.process_single_image(
                    image=input_source,
                    user_prompt=user_prompt,
                    use_layout=use_layout,
                    format_type=format_type
                )
            
            # Formater le resultat
            formatted_result = self._format_result(result, output_format)
            
            # Nettoyage memoire final
            if self.memory_manager.check_memory_limit():
                self.memory_manager.cleanup_memory()
            
            return formatted_result
            
        except Exception as e:
            logger.error(f"Pipeline processing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "text": "",
                "structured_data": None
            }
    
    def _format_result(self, result: ProcessingResult, output_format: str) -> Dict[str, Any]:
        """Formate le resultat selon le format demande"""
        base_result = {
            "success": result.success,
            "text": result.text,
            "structured_data": result.structured_data,
            "pages_processed": result.pages_processed
        }
        
        if not result.success:
            base_result["error"] = result.error_message
            return base_result
        
        if output_format == "html":
            base_result["html"] = self.renderer.render_to_html(
                result.text, result.structured_data
            )
        elif output_format == "markdown":
            base_result["markdown"] = self.renderer.render_to_markdown(
                result.text, result.structured_data
            )
        # Pour JSON, on retourne deje le bon format
        
        return base_result
    
    def process_simple(self, file_path: str, user_prompt: str) -> Dict[str, Any]:
        """
        Interface simplifiee : fichier + prompt -> JSON structure
        
        Args:
            file_path: Chemin vers le fichier
            user_prompt: Ce qu'on veut extraire
            
        Returns:
            JSON avec texte extrait et donnees structurees
        """
        return self.process(
            input_source=file_path,
            user_prompt=user_prompt,
            use_layout=True,
            format_type="plain",
            output_format="json"
        )
    
    def process_ocr_only(self, file_path: str, format_type: str = "plain") -> str:
        """
        OCR simple sans raisonnement
        
        Args:
            file_path: Chemin vers le fichier
            format_type: Format de sortie ("plain", "markdown", "latex")
            
        Returns:
            Texte extrait
        """
        result = self.process(
            input_source=file_path,
            user_prompt=None,
            use_layout=True,
            format_type=format_type,
            output_format="json"
        )
        
        return result.get("text", "")
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Retourne des informations sur l'usage memoire"""
        return {
            "current_memory_mb": self.memory_manager.get_memory_usage(),
            "max_memory_mb": self.memory_manager.max_memory_mb,
            "memory_usage_percent": (
                self.memory_manager.get_memory_usage() / 
                self.memory_manager.max_memory_mb * 100
            )
        }
    
    def unload_models(self):
        """Decharge tous les modeles pour liberer la memoire"""
        logger.info("Unloading all models")
        self.model_cache.unload_all()
        self.processor.unload_models()
        self.memory_manager.cleanup_memory()
    
    def __del__(self):
        """Cleanup automatique"""
        try:
            self.unload_models()
        except:
            pass

# Instance globale pour faciliter l'usage
pipeline = OCRPipeline()