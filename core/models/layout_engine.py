import cv2
import numpy as np
import torch
import gc
from PIL import Image
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging

from config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class Zone:
    """Represente une zone detectee dans le document"""
    bbox: List[int]  # [x1, y1, x2, y2]
    zone_type: str   # 'text', 'table', 'figure', 'formula', 'title'
    confidence: float
    image: Optional[np.ndarray] = None
    
    @property
    def area(self) -> int:
        """Calcule l'aire de la zone"""
        return (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])
    
    def crop_from_image(self, image: np.ndarray) -> np.ndarray:
        """Decoupe la zone de l'image"""
        x1, y1, x2, y2 = self.bbox
        return image[y1:y2, x1:x2]

class LayoutEngine:
    """Wrapper pour PP-DocLayout - Segmentation de documents"""
    
    def __init__(self, device: str = "auto"):
        self.model = None
        self.device = self._get_device(device)
        self.model_name = settings.LAYOUT_MODEL
        self.confidence_threshold = 0.5
        
        # Mapping des classes PP-DocLayout
        self.class_names = {
            0: 'text', 
            1: 'title', 
            2: 'figure', 
            3: 'table', 
            4: 'formula'
        }
        
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
        """Lazy loading du modele PP-DocLayout"""
        if self.model is not None:
            return
            
        # SIMPLIFICATION: GOT-OCR 2.0 gère déjà le layout en interne
        # On désactive la segmentation externe pour éviter les complications
        logger.info("Layout detection disabled - GOT-OCR 2.0 handles layout internally")
        self.model = "disabled"
        self.model_type = 'disabled'
    
    def _load_yolo_fallback(self):
        """Charge un modele YOLOv8 comme fallback"""
        try:
            from ultralytics import YOLO
            
            # Utiliser un modele YOLOv8 pre-entraene pour la detection de documents
            self.model = YOLO('yolov8n.pt')  # ou un modele specialise
            self.model_type = 'yolo'
            
        except ImportError:
            logger.error("Neither PaddlePaddle nor YOLOv8 available")
            raise
    
    def _get_config_path(self) -> str:
        """Retourne le chemin vers la config PP-DocLayout"""
        # e adapter selon l'installation
        return "configs/picodet/legacy_model/application/layout_analysis/picodet_lcnet_x1_0_fgd_layout.yml"
    
    def _get_weights_path(self) -> str:
        """Retourne le chemin vers les poids du modele"""
        # e adapter selon l'installation
        return "weights/picodet_lcnet_x1_0_fgd_layout_cdla_infer.pdparams"
    
    def segment(self, image: np.ndarray) -> List[Zone]:
        """
        Segmente un document en zones
        
        Args:
            image: Image numpy array (BGR)
            
        Returns:
            Liste des zones detectees
        """
        if self.model is None:
            self.load()
            
        # Fallback simple: retourner toute l'image comme une zone
        # GOT-OCR 2.0 gère la segmentation en interne de toute façon
        h, w = image.shape[:2]
        return [Zone(
            bbox=[0, 0, w, h],
            zone_type='text',
            confidence=1.0
        )]
    
    def _segment_with_paddledet(self, image: np.ndarray) -> List[Zone]:
        """Segmentation avec PP-DocLayout/PaddleDetection"""
        results = self.model.predict([image])
        zones = []
        
        for result in results:
            if 'bbox' not in result:
                continue
                
            bboxes = result['bbox']
            scores = result.get('score', [])
            labels = result.get('label', [])
            
            for i, bbox in enumerate(bboxes):
                if i < len(scores) and scores[i] >= self.confidence_threshold:
                    x1, y1, x2, y2 = map(int, bbox[:4])
                    confidence = float(scores[i])
                    label = int(labels[i]) if i < len(labels) else 0
                    
                    zone_type = self.class_names.get(label, 'text')
                    
                    zones.append(Zone(
                        bbox=[x1, y1, x2, y2],
                        zone_type=zone_type,
                        confidence=confidence
                    ))
        
        return zones
    
    def _segment_with_yolo(self, image: np.ndarray) -> List[Zone]:
        """Segmentation avec YOLOv8 (fallback)"""
        results = self.model(image)
        zones = []
        
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
                
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                
                # Mapping approximatif pour YOLO vers nos classes
                zone_type = self._map_yolo_class(class_id)
                
                zones.append(Zone(
                    bbox=[x1, y1, x2, y2],
                    zone_type=zone_type,
                    confidence=confidence
                ))
        
        return zones
    
    def _map_paddleocr_class(self, class_id: int) -> str:
        """Map les classes PaddleOCR vers nos types de zones"""
        # Mapping selon PP-DocLayout
        mapping = {
            0: 'text',
            1: 'title', 
            2: 'list',
            3: 'table',
            4: 'figure'
        }
        return mapping.get(class_id, 'text')
    
    def _map_yolo_class(self, class_id: int) -> str:
        """Map les classes YOLO vers nos types de zones (legacy)"""
        # Mapping approximatif - e adapter selon le modele utilise
        mapping = {
            0: 'text',    # person -> text (approximation)
            1: 'figure',  # bicycle -> figure
            2: 'table',   # car -> table
            # ... autres mappings
        }
        return mapping.get(class_id, 'text')
    
    def segment_reading_order(self, zones: List[Zone]) -> List[Zone]:
        """
        Ordonne les zones selon l'ordre de lecture
        (de haut en bas, de gauche e droite)
        """
        def reading_order_key(zone: Zone) -> Tuple[int, int]:
            x1, y1, x2, y2 = zone.bbox
            # Grouper par lignes approximatives, puis par position horizontale
            line_group = y1 // 50  # Tolerance de 50px pour une ligne
            return (line_group, x1)
        
        return sorted(zones, key=reading_order_key)
    
    def merge_adjacent_text_zones(self, zones: List[Zone], 
                                  distance_threshold: int = 20) -> List[Zone]:
        """
        Fusionne les zones de texte adjacentes
        """
        text_zones = [z for z in zones if z.zone_type == 'text']
        other_zones = [z for z in zones if z.zone_type != 'text']
        
        if len(text_zones) <= 1:
            return zones
        
        merged = []
        used = set()
        
        for i, zone1 in enumerate(text_zones):
            if i in used:
                continue
                
            # Trouver zones adjacentes
            adjacent = [zone1]
            used.add(i)
            
            for j, zone2 in enumerate(text_zones[i+1:], i+1):
                if j in used:
                    continue
                    
                if self._zones_adjacent(zone1, zone2, distance_threshold):
                    adjacent.append(zone2)
                    used.add(j)
            
            # Creer zone fusionnee
            if len(adjacent) > 1:
                merged_zone = self._merge_zones(adjacent)
                merged.append(merged_zone)
            else:
                merged.append(zone1)
        
        return merged + other_zones
    
    def _zones_adjacent(self, zone1: Zone, zone2: Zone, threshold: int) -> bool:
        """Verifie si deux zones sont adjacentes"""
        x1_1, y1_1, x2_1, y2_1 = zone1.bbox
        x1_2, y1_2, x2_2, y2_2 = zone2.bbox
        
        # Distance horizontale
        h_dist = min(abs(x2_1 - x1_2), abs(x2_2 - x1_1))
        # Distance verticale  
        v_dist = min(abs(y2_1 - y1_2), abs(y2_2 - y1_1))
        
        return h_dist <= threshold or v_dist <= threshold
    
    def _merge_zones(self, zones: List[Zone]) -> Zone:
        """Fusionne plusieurs zones en une"""
        if not zones:
            return None
            
        # Calculer bounding box englobante
        x1 = min(z.bbox[0] for z in zones)
        y1 = min(z.bbox[1] for z in zones)
        x2 = max(z.bbox[2] for z in zones)
        y2 = max(z.bbox[3] for z in zones)
        
        # Confiance moyenne ponderee par l'aire
        total_area = sum(z.area for z in zones)
        weighted_conf = sum(z.confidence * z.area for z in zones) / total_area
        
        return Zone(
            bbox=[x1, y1, x2, y2],
            zone_type=zones[0].zone_type,
            confidence=weighted_conf
        )
    
    def unload(self) -> None:
        """Decharge le modele pour liberer la memoire"""
        if self.model is not None:
            del self.model
            self.model = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            logger.info("Layout model unloaded")
    
    def __del__(self):
        """Cleanup automatique"""
        self.unload()