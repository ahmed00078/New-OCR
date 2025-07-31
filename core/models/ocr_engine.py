import torch
import gc
from PIL import Image
from typing import Optional
from transformers import AutoModelForImageTextToText, AutoProcessor
import logging

from config.settings import settings

logger = logging.getLogger(__name__)

class OCREngine:
    """Wrapper simple pour GOT-OCR 2.0"""
    
    def __init__(self, device: str = "auto"):
        self.model = None
        self.processor = None
        self.device = self._get_device(device)
        self.model_name = settings.OCR_MODEL
        
    def _get_device(self, device_preference: str) -> str:
        """Determine le device optimal"""
        if device_preference == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device_preference
    
    def load(self) -> None:
        """Lazy loading du modele GOT-OCR"""
        if self.model is not None:
            return
            
        try:
            logger.info(f"Loading GOT-OCR model: {self.model_name}")
            
            # Load processor (combinaison tokenizer + image processor)
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Load model avec la classe correcte pour GOT-OCR 2.0
            if self.device == "cuda":
                # Pour CUDA, utiliser device_map pour éviter les problèmes
                self.model = AutoModelForImageTextToText.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    device_map="auto"
                )
            else:
                # Pour CPU, chargement simple
                self.model = AutoModelForImageTextToText.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
                self.model = self.model.to(self.device)
            
            self.model.eval()
            logger.info(f"GOT-OCR model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load GOT-OCR model: {e}")
            raise
    
    def extract_text(self, image: Image.Image, format_type: str = "plain") -> str:
        """
        Extrait le texte d'une image
        
        Args:
            image: PIL Image
            format_type: "plain", "markdown", "latex"
        """
        if self.model is None:
            self.load()
            
        try:
            # Prepare image
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Préparer les inputs selon le format demandé
            if format_type == "markdown":
                inputs = self.processor(image, return_tensors="pt", format=True)
            elif format_type == "latex":
                inputs = self.processor(image, return_tensors="pt", format=True)
            else:
                inputs = self.processor(image, return_tensors="pt")
            
            # Move inputs to same device as model 
            if self.device == "cuda":
                # Pour CUDA, utiliser le device du modèle
                model_device = next(self.model.parameters()).device
                inputs = {k: v.to(model_device) for k, v in inputs.items() if hasattr(v, 'to')}
            else:
                # Pour CPU
                inputs = {k: v.to(self.device) for k, v in inputs.items() if hasattr(v, 'to')}
            
            # Générer le texte avec torch.no_grad pour économiser la mémoire
            with torch.no_grad():
                generate_ids = self.model.generate(
                    **inputs,
                    do_sample=False,
                    tokenizer=self.processor.tokenizer,
                    stop_strings="<|im_end|>",
                    max_new_tokens=4096,
                )
            
            # Décoder le résultat
            result = self.processor.decode(
                generate_ids[0, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            
            return result.strip()
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return ""
    
    def extract_with_boxes(self, image: Image.Image) -> dict:
        """
        Extrait le texte avec les coordonnees des boetes
        
        Returns:
            dict avec 'text' et 'boxes'
        """
        if self.model is None:
            self.load()
            
        try:
            # GOT-OCR avec coordonnees
            result = self.model.chat(
                self.tokenizer, 
                image, 
                ocr_type='ocr',
                ocr_box='<box>'
            )
            
            # Parse result pour extraire texte et coordonnees
            text, boxes = self._parse_box_result(result)
            
            return {
                'text': text,
                'boxes': boxes
            }
            
        except Exception as e:
            logger.error(f"OCR with boxes failed: {e}")
            return {'text': '', 'boxes': []}
    
    def _parse_box_result(self, result: str) -> tuple[str, list]:
        """Parse le resultat avec coordonnees"""
        # Implementation simplifiee - e ameliorer selon le format exact de GOT-OCR
        lines = result.split('\n')
        text_parts = []
        boxes = []
        
        for line in lines:
            if '<box>' in line and '</box>' in line:
                # Extraire coordonnees et texte
                try:
                    box_start = line.find('<box>')
                    box_end = line.find('</box>') + 6
                    box_content = line[box_start:box_end]
                    text_content = line[box_end:].strip()
                    
                    # Parse coordinates (format e adapter selon GOT-OCR)
                    coords = self._extract_coordinates(box_content)
                    if coords and text_content:
                        boxes.append({
                            'text': text_content,
                            'coordinates': coords
                        })
                        text_parts.append(text_content)
                except:
                    text_parts.append(line)
            else:
                text_parts.append(line)
        
        return '\n'.join(text_parts), boxes
    
    def _extract_coordinates(self, box_content: str) -> Optional[list]:
        """Extrait les coordonnees du format box"""
        # e implementer selon le format exact de GOT-OCR
        # Exemple: <box>x1,y1,x2,y2</box>
        try:
            coords_str = box_content.replace('<box>', '').replace('</box>', '')
            coords = [float(x.strip()) for x in coords_str.split(',')]
            return coords if len(coords) == 4 else None
        except:
            return None
    
    def unload(self) -> None:
        """Decharge le modele pour liberer la memoire"""
        if self.model is not None:
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            logger.info("GOT-OCR model unloaded")
    
    def __del__(self):
        """Cleanup automatique"""
        self.unload()