import torch
import gc
import json
from typing import Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

from config.settings import settings

logger = logging.getLogger(__name__)

class ReasoningEngine:
    """Wrapper pour AI - Extrait des informations structurees en JSON"""
    
    def __init__(self, device: str = "auto"):
        self.model = None
        self.tokenizer = None
        self.device = self._get_device(device)
        self.model_name = settings.REASONING_MODEL
        self.max_tokens = settings.MAX_TOKENS
        
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
        """Lazy loading du modele AI"""
        if self.model is not None:
            return
            
        try:
            logger.info(f"Loading AI model: {self.model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                trust_remote_code=True
            )
            
            # Configuration avec quantization 8-bit si GPU disponible
            model_kwargs = {
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
                "device_map": "auto" if self.device == "cuda" else None
            }
            
            # Ajouter quantization si GPU disponible et bitsandbytes installé
            if self.device == "cuda":
                try:
                    from transformers import BitsAndBytesConfig
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_threshold=6.0,
                        llm_int8_enable_fp32_cpu_offload=False
                    )
                    logger.info("Using 8-bit quantization for AI model")
                except ImportError:
                    logger.warning("bitsandbytes not available, using full precision")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            # Move to device if not using device_map
            if self.device != "cuda":
                self.model = self.model.to(self.device)
            
            self.model.eval()
            logger.info(f"AI model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load AI model: {e}")
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
        ocr_text = self._truncate_text(ocr_text)
        
        # Creer le prompt pour extraction JSON
        prompt = self._create_json_prompt(ocr_text, user_prompt)
        
        try:
            response = self._generate_response(prompt)
            json_result = self._parse_json_response(response)
            return json_result
            
        except Exception as e:
            logger.error(f"JSON extraction failed: {e}")
            return {"error": "Extraction failed", "details": str(e)}
    
    def _create_json_prompt(self, ocr_text: str, user_prompt: str) -> str:
        """Cree un prompt optimise pour generer du JSON selon le template du repo original"""
        return f"""<|im_start|>system
You are an intelligent data extraction assistant. Analyze the text according to the instructions.
Respond ONLY with valid JSON.
<|im_end|>
<|im_start|>user
Text to analyze:
{ocr_text}

Extraction instructions:
{user_prompt}

RULES:
    - Use null if information is not available
    - Return ONLY valid JSON, no markdown formatting
    - Do not include any additional text or explanations

JSON Response:
<|im_end|>
<|im_start|>assistant"""
    
    def _generate_response(self, prompt: str, max_new_tokens: int = 512) -> str:
        """Genere une reponse avec le modele"""
        try:
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True,
                max_length=self.max_tokens - max_new_tokens
            )
            
            if torch.cuda.is_available() and self.device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            input_length = inputs['input_ids'].shape[1]
            response_tokens = outputs[0][input_length:]
            response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return "{}"
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse la reponse pour extraire le JSON valide"""
        try:
            # Nettoyer la reponse
            response = response.strip()
            
            # Trouver le JSON dans la reponse
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
            else:
                # Fallback: essayer de parser toute la reponse
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
        """Decharge le modele pour liberer la memoire"""
        if self.model is not None:
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            logger.info("AI model unloaded")
    
    def __del__(self):
        """Cleanup automatique"""
        self.unload()