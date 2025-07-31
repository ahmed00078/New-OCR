import os
from typing import Optional
from dataclasses import dataclass

@dataclass
class Settings:
    # Models
    OCR_MODEL: str = "stepfun-ai/GOT-OCR-2.0-hf"
    LAYOUT_MODEL: str = "PP-DocLayout_plus-L"
    REASONING_MODEL: str = "Qwen/Qwen2.5-7B-Instruct"
    
    # Performance
    MAX_MEMORY_MB: int = 20000
    BATCH_SIZE: int = 5
    MAX_PAGES: int = 50
    MAX_TOKENS: int = 4096
    
    # Paths
    TEMP_DIR: str = "/tmp/got-ocr"
    
    # Device
    DEVICE: str = "auto"  # auto, cpu, cuda
    
    # API
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    @classmethod
    def from_env(cls) -> 'Settings':
        """Override settings from environment variables"""
        return cls(
            OCR_MODEL=os.getenv("OCR_MODEL", cls.OCR_MODEL),
            LAYOUT_MODEL=os.getenv("LAYOUT_MODEL", cls.LAYOUT_MODEL),
            REASONING_MODEL=os.getenv("REASONING_MODEL", cls.REASONING_MODEL),
            MAX_MEMORY_MB=int(os.getenv("MAX_MEMORY_MB", cls.MAX_MEMORY_MB)),
            BATCH_SIZE=int(os.getenv("BATCH_SIZE", cls.BATCH_SIZE)),
            MAX_PAGES=int(os.getenv("MAX_PAGES", cls.MAX_PAGES)),
            MAX_TOKENS=int(os.getenv("MAX_TOKENS", cls.MAX_TOKENS)),
            TEMP_DIR=os.getenv("TEMP_DIR", cls.TEMP_DIR),
            DEVICE=os.getenv("DEVICE", cls.DEVICE),
            HOST=os.getenv("HOST", cls.HOST),
            PORT=int(os.getenv("PORT", cls.PORT)),
        )

# Global settings instance
settings = Settings.from_env()