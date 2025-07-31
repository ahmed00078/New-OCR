from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from enum import Enum

class ProcessingMode(str, Enum):
    """Modes de traitement disponibles"""
    OCR_ONLY = "ocr"
    EXTRACT = "extract"

class FormatType(str, Enum):
    """Formats de sortie OCR"""
    PLAIN = "plain"
    MARKDOWN = "markdown"
    LATEX = "latex"

class OutputFormat(str, Enum):
    """Formats de reponse"""
    JSON = "json"
    HTML = "html"
    MARKDOWN = "markdown"

class ProcessRequest(BaseModel):
    """Requete de traitement de document"""
    mode: ProcessingMode = Field(
        default=ProcessingMode.OCR_ONLY,
        description="Mode de traitement : 'ocr' pour extraction simple, 'extract' pour raisonnement"
    )
    user_prompt: Optional[str] = Field(
        default=None,
        description="Instructions pour extraction d'informations (requis pour mode 'extract')"
    )
    use_layout: bool = Field(
        default=True,
        description="Utiliser la segmentation de layout pour ameliorer l'OCR"
    )
    format_type: FormatType = Field(
        default=FormatType.PLAIN,
        description="Format de sortie OCR"
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.JSON,
        description="Format de la reponse"
    )

class ProcessResponse(BaseModel):
    """Reponse de traitement"""
    success: bool = Field(description="Statut du traitement")
    text: str = Field(description="Texte extrait par OCR")
    structured_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Donnees structurees extraites (si mode extract)"
    )
    pages_processed: int = Field(
        default=0,
        description="Nombre de pages traitees"
    )
    processing_time: Optional[float] = Field(
        default=None,
        description="Temps de traitement en secondes"
    )
    error: Optional[str] = Field(
        default=None,
        description="Message d'erreur si echec"
    )
    # Formats optionnels
    html: Optional[str] = Field(default=None, description="Rendu HTML")
    markdown: Optional[str] = Field(default=None, description="Rendu Markdown")

class HealthResponse(BaseModel):
    """Reponse de sante du service"""
    status: str
    memory_usage_mb: float
    models_loaded: List[str]
    version: str = "1.0.0"

class ErrorResponse(BaseModel):
    """Reponse d'erreur standard"""
    error: str
    detail: Optional[str] = None
    code: Optional[int] = None