import os
import time
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging

from api.models import (
    ProcessRequest, ProcessResponse, HealthResponse, ErrorResponse,
    ProcessingMode, FormatType, OutputFormat
)
from core.pipeline import pipeline
from config.settings import settings

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Creation de l'app FastAPI
app = FastAPI(
    title="GOT-OCR Pipeline API",
    description="API simplifiee pour GOT-OCR 2.0 avec raisonnement et segmentation",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_model=dict)
async def root():
    """Endpoint racine"""
    return {
        "message": "GOT-OCR Pipeline API",
        "version": "1.0.0",
        "endpoints": {
            "process": "/process",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Verification de sante du service"""
    try:
        memory_info = pipeline.get_memory_info()
        
        return HealthResponse(
            status="healthy",
            memory_usage_mb=memory_info["current_memory_mb"],
            models_loaded=[]  # e implementer si necessaire
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process", response_model=ProcessResponse)
async def process_document(
    file: UploadFile = File(...),
    mode: ProcessingMode = Form(default=ProcessingMode.OCR_ONLY),
    user_prompt: str = Form(default=None),
    use_layout: bool = Form(default=True),
    format_type: FormatType = Form(default=FormatType.PLAIN),
    output_format: OutputFormat = Form(default=OutputFormat.JSON)
):
    """
    Traite un document uploade
    
    Args:
        file: Fichier e traiter (PDF, JPG, PNG, etc.)
        mode: Mode de traitement ('ocr' ou 'extract')
        user_prompt: Instructions d'extraction (requis pour mode 'extract')
        use_layout: Utiliser la segmentation de layout
        format_type: Format OCR de sortie
        output_format: Format de la reponse
    """
    start_time = time.time()
    temp_file_path = None
    
    try:
        # Validation
        if mode == ProcessingMode.EXTRACT and not user_prompt:
            raise HTTPException(
                status_code=400,
                detail="user_prompt is required when mode is 'extract'"
            )
        
        # Verifier le type de fichier
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        file_ext = os.path.splitext(file.filename)[1].lower()
        allowed_extensions = ['.pdf', '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file_ext}. Allowed: {allowed_extensions}"
            )
        
        # Sauvegarder le fichier temporairement
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            temp_file_path = temp_file.name
            content = await file.read()
            temp_file.write(content)
        
        logger.info(f"Processing file: {file.filename} ({len(content)} bytes)")
        
        # Traitement selon le mode
        if mode == ProcessingMode.EXTRACT:
            result = pipeline.process(
                input_source=temp_file_path,
                user_prompt=user_prompt,
                use_layout=use_layout,
                format_type=format_type.value,
                output_format=output_format.value
            )
        else:
            result = pipeline.process(
                input_source=temp_file_path,
                user_prompt=None,
                use_layout=use_layout,
                format_type=format_type.value,
                output_format=output_format.value
            )
        
        # Calculer le temps de traitement
        processing_time = time.time() - start_time
        
        # Construire la reponse
        response = ProcessResponse(
            success=result["success"],
            text=result["text"],
            structured_data=result.get("structured_data"),
            pages_processed=result.get("pages_processed", 0),
            processing_time=processing_time,
            error=result.get("error"),
            html=result.get("html"),
            markdown=result.get("markdown")
        )
        
        logger.info(f"Processing completed in {processing_time:.2f}s")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        processing_time = time.time() - start_time
        
        return ProcessResponse(
            success=False,
            text="",
            structured_data=None,
            pages_processed=0,
            processing_time=processing_time,
            error=str(e)
        )
    finally:
        # Nettoyage du fichier temporaire
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file: {e}")

@app.post("/process-simple")
async def process_simple(
    file: UploadFile = File(...),
    user_prompt: str = Form(...)
):
    """
    Interface simplifiee : fichier + prompt -> JSON structure
    
    Args:
        file: Fichier e traiter
        user_prompt: Instructions d'extraction
    """
    return await process_document(
        file=file,
        mode=ProcessingMode.EXTRACT,
        user_prompt=user_prompt,
        use_layout=True,
        format_type=FormatType.PLAIN,
        output_format=OutputFormat.JSON
    )

@app.get("/memory")
async def get_memory_info():
    """Informations sur l'usage memoire"""
    try:
        return pipeline.get_memory_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/unload-models")
async def unload_models():
    """Decharge tous les modeles pour liberer la memoire"""
    try:
        pipeline.unload_models()
        return {"message": "Models unloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Gestionnaire d'erreurs global
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc)
        ).dict()
    )

if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting server on {settings.HOST}:{settings.PORT}")
    uvicorn.run(
        app,
        host=settings.HOST,
        port=settings.PORT,
        log_level="info"
    )