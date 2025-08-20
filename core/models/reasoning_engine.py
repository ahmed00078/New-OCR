import torch
import gc
import json
from typing import Dict, Any
import logging
import os
from pathlib import Path

from config.settings import settings

logger = logging.getLogger(__name__)

class ReasoningEngine:
    """Wrapper pour AI - Extrait des informations structurees en JSON"""
    
    def __init__(self, device: str = "auto"):
        self.model = None
        self.device = self._get_device(device)
        self.model_name = settings.REASONING_MODEL
        self.max_tokens = settings.MAX_TOKENS
        self.gguf_model_path = "gpt-oss-20b-Q4_0.gguf"
        
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
        """Lazy loading du modele GGUF"""
        if self.model is not None:
            return
            
        try:
            logger.info(f"Loading GGUF model: {self.gguf_model_path}")
            
            # Importer llama-cpp-python
            try:
                from llama_cpp import Llama
            except ImportError:
                logger.error("llama-cpp-python not installed. Install with: pip install llama-cpp-python")
                raise ImportError("llama-cpp-python is required for GGUF models")
            
            # Télécharger le modèle GGUF si nécessaire
            model_path = self._download_gguf_model()
            
            # Configuration pour GGUF - Limiter les couches GPU pour éviter OOM
            n_gpu_layers = 30 if self.device == "cuda" else 0  # Seulement 30 couches sur GPU
            
            self.model = Llama(
                model_path=model_path,
                n_gpu_layers=n_gpu_layers,
                n_ctx=self.max_tokens,
                verbose=False,
                chat_format="chatml"  # Format compatible avec GPT-OSS
            )
            
            logger.info(f"GGUF model loaded successfully with {n_gpu_layers} GPU layers")
            
        except Exception as e:
            logger.error(f"Failed to load GGUF model: {e}")
            raise
    
    def _download_gguf_model(self) -> str:
        """Télécharge le modèle GGUF depuis Hugging Face"""
        try:
            from huggingface_hub import hf_hub_download
            
            cache_dir = Path.home() / ".cache" / "huggingface" / "gguf"
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            model_path = hf_hub_download(
                repo_id=self.model_name,
                filename=self.gguf_model_path,
                cache_dir=cache_dir
            )
            
            logger.info(f"GGUF model downloaded to: {model_path}")
            return model_path
            
        except Exception as e:
            logger.error(f"Failed to download GGUF model: {e}")
            raise
    
    def extract_to_json(self, ocr_text: str, user_prompt: str) -> Dict[str, Any]:
        """
        Transforme le texte OCR en JSON selon les besoins de l'utilisateur
        
        Args:
            ocr_text: Texte extrait par GOT-OCR
            user_prompt: Instructions de l'utilisateur (ex: "extraire nom, ege, adresse")
            
        Returns:
            Dictionnaire JSON avec les informations extraites
        """
        if self.model is None:
            self.load()
            
        # Tronquer le texte si trop long
        # ocr_text = self._truncate_text(ocr_text)  # This was not commanted
        
        # Creer le prompt pour extraction JSON
        prompt = self._create_json_prompt(ocr_text, user_prompt)
        
        # DEBUG: Afficher le prompt complet envoyé au modèle
        logger.info("=" * 80)
        logger.info("PROMPT COMPLET ENVOYÉ AU MODÈLE:")
        logger.info("=" * 80)
        logger.info(prompt)
        logger.info("=" * 80)
        
        try:
            response = self._generate_response(prompt)
            
            # DEBUG: Afficher la réponse brute du modèle
            logger.info("RÉPONSE BRUTE DU MODÈLE:")
            logger.info("-" * 80)
            logger.info(response)
            logger.info("-" * 80)
            
            json_result = self._parse_json_response(response)
            return json_result
            
        except Exception as e:
            logger.error(f"JSON extraction failed: {e}")
            return {"error": "Extraction failed", "details": str(e)}
    
    def _create_json_prompt(self, ocr_text: str, user_prompt: str) -> str:
        """Cree un prompt optimise pour generer du JSON selon le template du repo original"""
        return f"""<|im_start|>system
You are a data extraction expert. Extract information from text and return ONLY a valid JSON object.
<|im_end|>
<|im_start|>user
Extract the following information from this document text:

{user_prompt}

Document text:
{ocr_text}

Instructions:
- Return ONLY a JSON object with the requested fields
- Use null for missing information
- Do not add explanations or markdown
- Start your response with {{ and end with }}

<|im_end|>
<|im_start|>assistant
{{"""
    
    def _generate_response(self, prompt: str, max_new_tokens: int = 1024) -> str:       # was max_new_tokens: int = 512
        """Genere une reponse avec le modele GGUF"""
        try:
            # Créer le message au format ChatML pour GPT-OSS
            messages = [
                {"role": "system", "content": "You are a data extraction expert. Extract information from text and return ONLY a valid JSON object."},
                {"role": "user", "content": prompt}
            ]
            
            response = self.model.create_chat_completion(
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=0.1,
                repeat_penalty=1.2,
                top_p=0.95,
                stop=["<|im_end|>", "</s>"]
            )
            
            # Extraire le contenu de la réponse
            content = response['choices'][0]['message']['content']
            return content.strip()
            
        except Exception as e:
            logger.error(f"GGUF response generation failed: {e}")
            return "{}"
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse la reponse pour extraire le JSON valide"""
        try:
            # Nettoyer la reponse
            response = response.strip()
            
            # Chercher les patterns possibles de JSON
            import re
            
            # Pattern pour trouver JSON dans des balises
            json_patterns = [
                r'<\|message\|>({.*?})',  # Pattern GPT-OSS
                r'```json\s*({.*?})\s*```',  # Pattern markdown
                r'({.*?})',  # Pattern général
            ]
            
            for pattern in json_patterns:
                matches = re.findall(pattern, response, re.DOTALL)
                for match in matches:
                    try:
                        return json.loads(match.strip())
                    except:
                        continue
            
            # Fallback: chercher manuellement le JSON
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
            else:
                # Dernier fallback: essayer de parser toute la reponse
                return json.loads(response)
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            logger.error(f"Response was: {response}")
            
            # Retourner un JSON d'erreur avec le texte brut
            return {
                "error": "Format JSON invalide",
                "raw_response": response,
                "parsing_error": str(e)
            }
        except Exception as e:
            logger.error(f"Unexpected error in JSON parsing: {e}")
            return {
                "error": "Erreur inattendue",
                "details": str(e)
            }
    
    def _truncate_text(self, text: str) -> str:
        """Tronque le texte pour respecter les limites de contexte"""
        if not text:
            return ""
            
        max_chars = (self.max_tokens - 300) * 4  # Reserve 300 tokens pour le prompt
        
        if len(text) <= max_chars:
            return text
            
        # Troncature intelligente
        truncated = text[:max_chars]
        
        # Essayer de couper e une ligne complete
        last_newline = truncated.rfind('\n')
        if last_newline > max_chars * 0.8:
            return truncated[:last_newline]
        else:
            return truncated + "..."
    
    def unload(self) -> None:
        """Decharge le modele GGUF pour liberer la memoire"""
        if self.model is not None:
            del self.model
            self.model = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            logger.info("GGUF model unloaded")
    
    def __del__(self):
        """Cleanup automatique"""
        self.unload()