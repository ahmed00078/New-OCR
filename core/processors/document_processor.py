import os
import logging
from typing import List, Dict, Any, Optional, Union
from PIL import Image
import numpy as np
from dataclasses import dataclass

from core.models.ocr_engine import OCREngine
from core.models.layout_engine import LayoutEngine, Zone
from core.models.reasoning_engine import ReasoningEngine
from core.processors.pdf_handler import PDFHandler
from config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class ProcessingResult:
    """Resultat du traitement d'un document"""
    text: str
    structured_data: Optional[Dict[str, Any]] = None
    zones: Optional[List[Zone]] = None
    pages_processed: int = 0
    success: bool = True
    error_message: Optional[str] = None

class DocumentProcessor:
    """Orchestrateur principal pour le traitement de documents"""
    
    def __init__(self):
        self.ocr_engine = OCREngine()
        self.layout_engine = LayoutEngine()
        self.reasoning_engine = ReasoningEngine()
        self.pdf_handler = PDFHandler()
        
    def process_document(self,
                        file_path: str,
                        user_prompt: Optional[str] = None,
                        use_layout: bool = True,
                        format_type: str = "plain") -> ProcessingResult:
        """
        Traite un document complet
        
        Args:
            file_path: Chemin vers le fichier
            user_prompt: Instructions utilisateur pour extraction JSON
            use_layout: Utiliser la segmentation de layout
            format_type: Format OCR ("plain", "markdown", "latex")
            
        Returns:
            ProcessingResult avec texte et donnees structurees
        """
        try:
            logger.info(f"Processing document: {file_path}")
            
            # 1. Preparer les images
            images = self._prepare_images(file_path)
            if not images:
                return ProcessingResult(
                    text="",
                    success=False,
                    error_message="Impossible de charger les images du document"
                )
            
            logger.info(f"Processing {len(images)} pages")
            
            # 2. Traitement par batch pour economiser la memoire
            all_text_parts = []
            all_zones = []
            
            batch_size = settings.BATCH_SIZE
            for i in range(0, len(images), batch_size):
                batch = images[i:i + batch_size]
                
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(images)-1)//batch_size + 1}")
                
                batch_text, batch_zones = self._process_image_batch(
                    batch, use_layout, format_type
                )
                
                all_text_parts.extend(batch_text)
                all_zones.extend(batch_zones)
            
            # 3. Combiner tous les textes
            full_text = self._combine_texts(all_text_parts)
            
            # 4. Extraction d'informations si demandee
            structured_data = None
            if user_prompt and full_text.strip():
                logger.info("Extracting structured information")
                structured_data = self.reasoning_engine.extract_to_json(
                    full_text, user_prompt
                )
            
            return ProcessingResult(
                text=full_text,
                structured_data=structured_data,
                zones=all_zones,
                pages_processed=len(images),
                success=True
            )
            
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            return ProcessingResult(
                text="",
                success=False,
                error_message=str(e)
            )
    
    def _prepare_images(self, file_path: str) -> List[Image.Image]:
        """Convertit le fichier en images PIL"""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext == '.pdf':
                # Convertir PDF en images
                return self.pdf_handler.pdf_to_images(file_path)
            elif file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']:
                # Charger image unique
                image = Image.open(file_path)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                return [image]
            else:
                raise ValueError(f"Format de fichier non supporte: {file_ext}")
                
        except Exception as e:
            logger.error(f"Failed to prepare images from {file_path}: {e}")
            return []
    
    def _process_image_batch(self, 
                           images: List[Image.Image], 
                           use_layout: bool, 
                           format_type: str) -> tuple[List[str], List[Zone]]:
        """Traite un batch d'images"""
        batch_texts = []
        batch_zones = []
        
        for image in images:
            try:
                if use_layout:
                    # Segmentation puis OCR par zone
                    text, zones = self._process_with_layout(image, format_type)
                else:
                    # OCR direct de l'image entiere
                    text = self.ocr_engine.extract_text(image, format_type)
                    zones = []
                
                batch_texts.append(text)
                batch_zones.extend(zones)
                
            except Exception as e:
                logger.error(f"Failed to process image: {e}")
                batch_texts.append("")
        
        return batch_texts, batch_zones
    
    def _process_with_layout(self, 
                           image: Image.Image, 
                           format_type: str) -> tuple[str, List[Zone]]:
        """Traite une image avec segmentation de layout"""
        try:
            # Convertir PIL en numpy pour layout engine
            image_np = np.array(image)
            
            # Segmenter le document
            zones = self.layout_engine.segment(image_np)
            
            if not zones:
                # Fallback: traiter l'image entiere
                text = self.ocr_engine.extract_text(image, format_type)
                return text, []
            
            # Ordonner les zones selon l'ordre de lecture
            zones = self.layout_engine.segment_reading_order(zones)
            
            # OCR de chaque zone
            zone_texts = []
            for zone in zones:
                try:
                    # Decouper la zone de l'image
                    zone_image = zone.crop_from_image(image_np)
                    zone_pil = Image.fromarray(zone_image)
                    
                    # OCR de la zone
                    zone_text = self.ocr_engine.extract_text(zone_pil, format_type)
                    
                    if zone_text.strip():
                        # Ajouter un marqueur du type de zone si necessaire
                        if format_type == "markdown" and zone.zone_type == "title":
                            zone_text = f"# {zone_text}"
                        elif format_type == "markdown" and zone.zone_type == "table":
                            zone_text = f"\n{zone_text}\n"
                        
                        zone_texts.append(zone_text)
                        
                except Exception as e:
                    logger.error(f"Failed to process zone {zone.zone_type}: {e}")
                    continue
            
            # Combiner les textes des zones
            full_text = self._combine_zone_texts(zone_texts, format_type)
            
            return full_text, zones
            
        except Exception as e:
            logger.error(f"Layout processing failed: {e}")
            # Fallback: OCR direct
            text = self.ocr_engine.extract_text(image, format_type)
            return text, []
    
    def _combine_zone_texts(self, zone_texts: List[str], format_type: str) -> str:
        """Combine les textes des zones selon le format"""
        if not zone_texts:
            return ""
        
        if format_type == "markdown":
            # Separateur plus naturel pour Markdown
            return "\n\n".join(text.strip() for text in zone_texts if text.strip())
        elif format_type == "latex":
            # Separateur approprie pour LaTeX
            return "\n\\par\n".join(text.strip() for text in zone_texts if text.strip())
        else:
            # Format plain text
            return "\n\n".join(text.strip() for text in zone_texts if text.strip())
    
    def _combine_texts(self, text_parts: List[str]) -> str:
        """Combine les textes de toutes les pages"""
        if not text_parts:
            return ""
            
        # Filtrer les textes vides et combiner avec separateur de page
        valid_texts = [text.strip() for text in text_parts if text.strip()]
        
        if not valid_texts:
            return ""
        
        return "\n\n--- Page suivante ---\n\n".join(valid_texts)
    
    def process_single_image(self, 
                           image: Union[str, Image.Image],
                           user_prompt: Optional[str] = None,
                           use_layout: bool = True,
                           format_type: str = "plain") -> ProcessingResult:
        """
        Traite une seule image
        
        Args:
            image: Chemin vers l'image ou objet PIL Image
            user_prompt: Instructions utilisateur
            use_layout: Utiliser la segmentation
            format_type: Format de sortie
        """
        try:
            # Charger l'image si c'est un chemin
            if isinstance(image, str):
                pil_image = Image.open(image)
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
            else:
                pil_image = image
            
            # Traiter l'image
            if use_layout:
                text, zones = self._process_with_layout(pil_image, format_type)
            else:
                text = self.ocr_engine.extract_text(pil_image, format_type)
                zones = []
            
            # Extraction d'informations si demandee
            structured_data = None
            if user_prompt and text.strip():
                structured_data = self.reasoning_engine.extract_to_json(
                    text, user_prompt
                )
            
            return ProcessingResult(
                text=text,
                structured_data=structured_data,
                zones=zones,
                pages_processed=1,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Single image processing failed: {e}")
            return ProcessingResult(
                text="",
                success=False,
                error_message=str(e)
            )
    
    def unload_models(self):
        """Decharge tous les modeles pour liberer la memoire"""
        logger.info("Unloading all models")
        self.ocr_engine.unload()
        self.layout_engine.unload()
        self.reasoning_engine.unload()
    
    def __del__(self):
        """Cleanup automatique"""
        self.unload_models()