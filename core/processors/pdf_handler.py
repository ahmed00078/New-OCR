import os
import tempfile
from typing import List, Optional
from PIL import Image
import logging

from config.settings import settings

logger = logging.getLogger(__name__)

class PDFHandler:
    """Gestionnaire pour les fichiers PDF avec gestion memoire optimisee"""
    
    def __init__(self):
        self.dpi = 150  # DPI pour conversion PDF -> image
        self.max_pages = settings.MAX_PAGES
        
    def pdf_to_images(self, pdf_path: str, start_page: int = 0, end_page: Optional[int] = None) -> List[Image.Image]:
        """
        Convertit un PDF en liste d'images PIL
        
        Args:
            pdf_path: Chemin vers le fichier PDF
            start_page: Page de debut (0-indexe)
            end_page: Page de fin (None = toutes les pages)
            
        Returns:
            Liste d'images PIL
        """
        try:
            # Essayer pdf2image en premier (meilleure qualite)
            return self._convert_with_pdf2image(pdf_path, start_page, end_page)
            
        except ImportError:
            logger.warning("pdf2image not available, trying PyMuPDF")
            try:
                return self._convert_with_pymupdf(pdf_path, start_page, end_page)
            except ImportError:
                logger.warning("PyMuPDF not available, trying pypdf")
                return self._convert_with_pypdf(pdf_path, start_page, end_page)
    
    def _convert_with_pdf2image(self, pdf_path: str, start_page: int, end_page: Optional[int]) -> List[Image.Image]:
        """Conversion avec pdf2image (recommande)"""
        from pdf2image import convert_from_path
        
        logger.info(f"Converting PDF with pdf2image: {pdf_path}")
        
        # Calculer les pages e traiter
        if end_page is None:
            end_page = min(start_page + self.max_pages, self._get_pdf_page_count(pdf_path))
        
        first_page = start_page + 1  # pdf2image utilise 1-indexe
        last_page = min(end_page, start_page + self.max_pages)
        
        images = convert_from_path(
            pdf_path,
            dpi=self.dpi,
            first_page=first_page,
            last_page=last_page,
            fmt='RGB'
        )
        
        logger.info(f"Converted {len(images)} pages")
        return images
    
    def _convert_with_pymupdf(self, pdf_path: str, start_page: int, end_page: Optional[int]) -> List[Image.Image]:
        """Conversion avec PyMuPDF (fitz) - alternative"""
        import fitz  # PyMuPDF
        
        logger.info(f"Converting PDF with PyMuPDF: {pdf_path}")
        
        doc = fitz.open(pdf_path)
        images = []
        
        if end_page is None:
            end_page = min(start_page + self.max_pages, len(doc))
        
        end_page = min(end_page, start_page + self.max_pages)
        
        for page_num in range(start_page, end_page):
            try:
                page = doc[page_num]
                
                # Creer matrice de transformation pour le DPI
                zoom = self.dpi / 72.0  # 72 DPI par defaut
                mat = fitz.Matrix(zoom, zoom)
                
                # Rendre la page en image
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("ppm")
                
                # Convertir en PIL Image
                import io
                image = Image.open(io.BytesIO(img_data))
                images.append(image)
                
            except Exception as e:
                logger.error(f"Failed to process page {page_num}: {e}")
                continue
        
        doc.close()
        logger.info(f"Converted {len(images)} pages")
        return images
    
    def _convert_with_pypdf(self, pdf_path: str, start_page: int, end_page: Optional[int]) -> List[Image.Image]:
        """Conversion avec pypdf - fallback basique"""
        logger.warning("Using pypdf fallback - may require system tools")
        
        try:
            return self._convert_with_system_tools(pdf_path, start_page, end_page)
        except Exception as e:
            logger.error(f"System tools conversion failed: {e}")
            raise ImportError("No suitable PDF conversion library available")
    
    def _convert_with_system_tools(self, pdf_path: str, start_page: int, end_page: Optional[int]) -> List[Image.Image]:
        """Utilise des outils systeme (pdftoppm) si disponibles"""
        import subprocess
        import glob
        
        if end_page is None:
            end_page = min(start_page + self.max_pages, self._get_pdf_page_count(pdf_path))
        
        # Creer un dossier temporaire
        with tempfile.TemporaryDirectory() as temp_dir:
            output_prefix = os.path.join(temp_dir, "page")
            
            # Essayer pdftoppm
            try:
                cmd = [
                    "pdftoppm",
                    "-jpeg",
                    "-f", str(start_page + 1),  # pdftoppm utilise 1-indexe
                    "-l", str(end_page),
                    "-r", str(self.dpi),
                    pdf_path,
                    output_prefix
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0:
                    # Charger les images generees
                    image_files = sorted(glob.glob(f"{output_prefix}-*.jpg"))
                    images = []
                    
                    for img_file in image_files:
                        try:
                            image = Image.open(img_file)
                            images.append(image.copy())
                        except Exception as e:
                            logger.error(f"Failed to load image {img_file}: {e}")
                    
                    return images
                    
            except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                logger.warning(f"pdftoppm failed: {e}")
        
        raise Exception("PDF conversion failed")
    
    def _get_pdf_page_count(self, pdf_path: str) -> int:
        """Obtient le nombre de pages du PDF"""
        try:
            # Essayer avec PyMuPDF
            import fitz
            doc = fitz.open(pdf_path)
            count = len(doc)
            doc.close()
            return count
            
        except ImportError:
            try:
                # Essayer avec pypdf
                from pypdf import PdfReader
                reader = PdfReader(pdf_path)
                return len(reader.pages)
                
            except ImportError:
                # Fallback: estimer avec la taille du fichier
                file_size = os.path.getsize(pdf_path)
                estimated_pages = max(1, file_size // (50 * 1024))  # ~50KB par page
                logger.warning(f"Could not determine page count, estimating {estimated_pages} pages")
                return estimated_pages
    
    def get_pdf_info(self, pdf_path: str) -> dict:
        """Retourne des informations sur le PDF"""
        try:
            page_count = self._get_pdf_page_count(pdf_path)
            file_size = os.path.getsize(pdf_path)
            
            return {
                "page_count": page_count,
                "file_size_mb": round(file_size / (1024 * 1024), 2),
                "estimated_processing_time": page_count * 2,  # ~2 secondes par page
                "max_processable_pages": self.max_pages
            }
            
        except Exception as e:
            logger.error(f"Failed to get PDF info: {e}")
            return {
                "page_count": 0,
                "file_size_mb": 0,
                "error": str(e)
            }