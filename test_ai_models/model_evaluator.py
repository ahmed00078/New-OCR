#!/usr/bin/env python3
"""
AI Model Tester for Environmental Data with Path References
Updated for the new dataset format using path-based extraction

Usage:
    python model_evaluator.py --model ollama:qwen3:8b --quick
    python model_evaluator.py --model transformers:microsoft/Phi-3.5-mini-instruct
    python model_evaluator.py --compare-responses
"""

import json
import time
import argparse
import os
import copy
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import re

# Only import what we need
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

@dataclass
class TestResult:
    """Test result with path-based evaluation"""
    document_id: int
    question_id: str
    model_name: str
    question: str
    predicted_answer: str
    expected_fields: List[str]
    expected_values: Dict
    processing_time: float
    field_scores: Dict[str, float]
    global_score: float
    error: Optional[str] = None

class ModelInterface:
    """Base interface for different model types"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.is_loaded = False
        
    def load_model(self):
        """Load the model"""
        raise NotImplementedError
        
    def generate_response(self, prompt: str) -> str:
        """Generate response from prompt"""
        raise NotImplementedError
        
    def cleanup(self):
        """Cleanup resources"""
        pass

class OllamaInterface(ModelInterface):
    """Interface for Ollama models"""
    
    def __init__(self, model_name: str, base_url: str = "http://localhost:11434"):
        super().__init__(model_name)
        self.base_url = base_url
        
    def load_model(self):
        """Check if Ollama is available and model exists"""            
        try:
            # Test Ollama connectivity
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code != 200:
                raise ConnectionError("Cannot connect to Ollama service")
                
            # Check if model is available
            models = response.json().get("models", [])
            model_names = [m["name"] for m in models]
            
            if self.model_name not in model_names:
                print(f"⚠️  Model {self.model_name} not found. Available models: {model_names}")
                print(f"🔄 Attempting to pull model...")
                
                # Try to pull the model
                pull_response = requests.post(
                    f"{self.base_url}/api/pull",
                    json={"name": self.model_name},
                    timeout=300
                )
                
                if pull_response.status_code != 200:
                    raise RuntimeError(f"Failed to pull model {self.model_name}")
                    
            self.is_loaded = True
            print(f"✅ Ollama model {self.model_name} ready")
            
        except Exception as e:
            raise RuntimeError(f"Ollama setup failed: {str(e)}")
    
    def generate_response(self, prompt: str) -> str:
        """Generate response using Ollama API"""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "top_p": 0.9,
                        "num_predict": 4096
                    }
                },
                timeout=300
            )
            
            if response.status_code == 200:
                return response.json().get("response", "").strip()
            else:
                raise RuntimeError(f"Ollama API error: {response.status_code}")
                
        except Exception as e:
            raise RuntimeError(f"Ollama generation failed: {str(e)}")

class TransformersInterface(ModelInterface):
    """Interface for Hugging Face Transformers models"""
    
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.tokenizer = None
        self.model = None
        self.device = None
        
    def load_model(self):
        """Load Transformers model"""            
        try:
            print(f"🔄 Loading Transformers model: {self.model_name}")
            
            # Determine device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"📱 Using device: {self.device}")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True
            )
            
            # Set pad token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            self.model.eval()
            self.is_loaded = True
            print(f"✅ Transformers model loaded")
            
        except Exception as e:
            raise RuntimeError(f"Transformers loading failed: {str(e)}")
    
    def generate_response(self, prompt: str) -> str:
        """Generate response using Transformers"""
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            return response.strip()
            
        except Exception as e:
            raise RuntimeError(f"Transformers generation failed: {str(e)}")
    
    def cleanup(self):
        """Cleanup Transformers resources"""
        if self.model is not None:
            del self.model
        if self.tokenizer is not None:
            del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

class PathBasedModelTester:
    """Model tester using path-based references"""
    
    def __init__(self, dataset_path: str = "dataset.json"):
        self.dataset = self._load_dataset(dataset_path)
        self.global_schema = self.dataset.get("global_schema", {})
        
    def _load_dataset(self, path: str) -> Dict:
        """Load the evaluation dataset"""
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        else:
            raise FileNotFoundError(f"Dataset file {path} not found")
    
    def create_model_interface(self, model_spec: str) -> ModelInterface:
        """Create appropriate model interface based on specification"""
        if model_spec.startswith("ollama:"):
            model_name = model_spec.split(":", 1)[1]
            return OllamaInterface(model_name)
        elif model_spec.startswith("transformers:"):
            model_name = model_spec.split(":", 1)[1]
            return TransformersInterface(model_name)
        else:
            raise ValueError(f"Unknown model specification: {model_spec}")
    
    def get_nested_value(self, obj: Dict, path: str):
        """Extract a value via path (ex: 'carbon_footprint.total.value')"""
        parts = path.split('.')
        current = obj
        
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        
        return current
    
    def set_nested_value(self, obj: Dict, path: str, value):
        """Set a value via path in nested dict"""
        parts = path.split('.')
        current = obj
        
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        current[parts[-1]] = value
    
    def create_minimal_schema(self, expected_fields: List[str]) -> Dict:
        """Create a minimal schema with only the expected fields"""
        minimal_schema = {}
        
        for field_path in expected_fields:
            parts = field_path.split('.')
            current = minimal_schema
            
            for i, part in enumerate(parts):
                if i == len(parts) - 1:
                    # Last part - this is where we want the value
                    current[part] = "<<EXTRACT_THIS>>"
                else:
                    # Intermediate part - create nested dict
                    if part not in current:
                        current[part] = {}
                    current = current[part]
        
        return minimal_schema
    
    def create_prompt(self, document_text: str, question_data: Dict) -> str:
        """Create prompt with minimal schema for specific question"""
        expected_fields = question_data["expected_fields"]
        question = question_data["question"]
        
        # Create minimal schema showing only expected fields
        minimal_schema = self.create_minimal_schema(expected_fields)
        
        prompt = f"""TASK: Extract specific information from this electronics document.

            DOCUMENT:
            {document_text}

            QUESTION: {question}

            EXTRACT ONLY THESE FIELDS:
            {', '.join(expected_fields)}

            OUTPUT FORMAT (return only the minimal JSON structure below):
            {json.dumps(minimal_schema, indent=2)}

            RULES:
            - Extract exact values and units from the document
            - Replace "<<EXTRACT_THIS>>" with the actual values found
            - Use null if information is not available
            - Return ONLY valid JSON, no markdown formatting
            - Keep the exact structure shown above

            JSON Response:"""

        # print(f"   📝 Generated Prompt:\n{prompt}\n")
        
        return prompt
    
    def calculate_field_score(self, extracted_value, expected_value) -> float:
        """Calculate score for a single field"""
        if extracted_value is None:
            return 0.0
        
        if expected_value is None:
            return 1.0 if extracted_value is not None else 0.0
        
        if isinstance(expected_value, dict):
            if not isinstance(extracted_value, dict):
                return 0.0
            
            # Compare each key in expected_value
            total_keys = len(expected_value)
            correct_keys = 0
            
            for key, exp_val in expected_value.items():
                ext_val = extracted_value.get(key)
                if self._values_match(ext_val, exp_val):
                    correct_keys += 1
            
            return correct_keys / total_keys if total_keys > 0 else 0.0
        
        elif isinstance(expected_value, list):
            if not isinstance(extracted_value, list):
                return 0.0
            return 1.0 if set(extracted_value) == set(expected_value) else 0.0
        
        else:
            return 1.0 if self._values_match(extracted_value, expected_value) else 0.0
    
    def _values_match(self, extracted, expected) -> bool:
        """Check if two values match (with some tolerance for numbers)"""
        if extracted == expected:
            return True
        
        # Handle number-string conversions in both directions
        def try_parse_number(value):
            """Try to parse a value as number"""
            if isinstance(value, (int, float)):
                return value
            if isinstance(value, str):
                try:
                    return float(value.strip()) if '.' in value else int(value.strip())
                except (ValueError, TypeError):
                    return None
            return None
        
        # Try numeric comparison
        expected_num = try_parse_number(expected)
        extracted_num = try_parse_number(extracted)
        
        if expected_num is not None and extracted_num is not None:
            return abs(extracted_num - expected_num) < 0.01
        
        # String comparison (case-insensitive)
        if isinstance(expected, str) and isinstance(extracted, str):
            return expected.lower().strip() == extracted.lower().strip()
        
        return False
    
    def _extract_field_values(self, predicted_answer: str, expected_fields: List[str]) -> Dict[str, Any]:
        """Extract only the values for expected fields from model response"""
        try:
            # Clean response (remove markdown and thinking tags if present)
            cleaned_response = predicted_answer.strip()
            
            # Handle DeepSeek-R1 thinking tags
            if "<think>" in cleaned_response and "</think>" in cleaned_response:
                # Extract content after </think>
                think_end = cleaned_response.find("</think>")
                cleaned_response = cleaned_response[think_end + len("</think>"):].strip()
            
            # Handle markdown code blocks
            if cleaned_response.startswith("```"):
                lines = cleaned_response.split('\n')
                cleaned_response = '\n'.join(lines[1:-1])
            
            response_json = json.loads(cleaned_response)
            
            # Extract only the expected field values
            extracted = {}
            for field_path in expected_fields:
                value = self.get_nested_value(response_json, field_path)
                if value is not None:
                    extracted[field_path] = value
            
            return extracted
            
        except (json.JSONDecodeError, Exception):
            return {}
    
    def evaluate_response(self, response_json: Dict, question_data: Dict) -> Dict:
        """Evaluate response using path matching"""
        expected_fields = question_data["expected_fields"]
        expected_values = question_data["expected_values"]
        
        field_scores = {}
        
        for field_path in expected_fields:
            # Extract the value from response
            extracted_value = self.get_nested_value(response_json, field_path)
            expected_value = expected_values.get(field_path)
            
            # Calculate score for this field
            field_score = self.calculate_field_score(extracted_value, expected_value)
            field_scores[field_path] = field_score
        
        # Global score = average of field scores
        global_score = sum(field_scores.values()) / len(field_scores) if field_scores else 0.0
        
        return {
            "global_score": global_score,
            "field_scores": field_scores,
            "total_fields": len(expected_fields),
            "extracted_fields": len([s for s in field_scores.values() if s > 0])
        }
    
    def test_model(
        self, 
        model_spec: str, 
        category_filter: Optional[str] = None,
        max_documents: Optional[int] = None
    ) -> List[TestResult]:
        """Test a model and return results with path-based evaluation"""
        
        print(f"\n🧪 TESTING MODEL: {model_spec}")
        print("=" * 50)
        
        # Create and load model
        try:
            model = self.create_model_interface(model_spec)
            model.load_model()
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            return []
        
        # Filter documents
        documents = self.dataset["documents"]
        if category_filter:
            documents = [d for d in documents if d["category"] == category_filter]
        if max_documents:
            documents = documents[:max_documents]
        
        print(f"📊 Testing on {len(documents)} documents")
        
        results = []
        
        for doc in documents:
            print(f"\n📄 Document {doc['id']}: {doc['title']}")
            print(f"   Category: {doc['category']}")
            
            for question_data in doc["questions"]:
                question_id = question_data["id"]
                question = question_data["question"]
                expected_fields = question_data["expected_fields"]
                expected_values = question_data["expected_values"]
                
                print(f"   ❓ {question_id}: {question}")
                print(f"      Expected fields: {len(expected_fields)}")
                
                # Generate prompt and get response
                prompt = self.create_prompt(doc["text"], question_data)
                
                start_time = time.time()
                try:
                    predicted = model.generate_response(prompt)
                    processing_time = time.time() - start_time

                    # print(f"   🤖 Predicted response: {predicted}")
                    
                    # Parse JSON response
                    try:
                        # Clean response (remove markdown and thinking tags if present)
                        cleaned_response = predicted.strip()
                        
                        # Handle DeepSeek-R1 thinking tags
                        if "<think>" in cleaned_response and "</think>" in cleaned_response:
                            # Extract content after </think>
                            think_end = cleaned_response.find("</think>")
                            cleaned_response = cleaned_response[think_end + len("</think>"):].strip()
                        
                        # Handle markdown code blocks
                        if cleaned_response.startswith("```"):
                            lines = cleaned_response.split('\n')
                            cleaned_response = '\n'.join(lines[1:-1])
                        
                        response_json = json.loads(cleaned_response)
                        
                        # Evaluate response
                        eval_result = self.evaluate_response(response_json, question_data)
                        
                        result = TestResult(
                            document_id=doc["id"],
                            question_id=question_id,
                            model_name=model_spec,
                            question=question,
                            predicted_answer=predicted,
                            expected_fields=expected_fields,
                            expected_values=expected_values,
                            processing_time=processing_time,
                            field_scores=eval_result["field_scores"],
                            global_score=eval_result["global_score"]
                        )
                        
                        print(f"      ⏱️  Time: {processing_time:.2f}s")
                        print(f"      📊 Score: {eval_result['global_score']:.2f} ({eval_result['extracted_fields']}/{eval_result['total_fields']} fields)")
                        
                    except json.JSONDecodeError as e:
                        result = TestResult(
                            document_id=doc["id"],
                            question_id=question_id,
                            model_name=model_spec,
                            question=question,
                            predicted_answer=predicted,
                            expected_fields=expected_fields,
                            expected_values=expected_values,
                            processing_time=processing_time,
                            field_scores={},
                            global_score=0.0,
                            error=f"JSON parsing error: {str(e)}"
                        )
                        print(f"      ❌ JSON Error: {e}")
                
                except Exception as e:
                    processing_time = time.time() - start_time
                    result = TestResult(
                        document_id=doc["id"],
                        question_id=question_id,
                        model_name=model_spec,
                        question=question,
                        predicted_answer="",
                        expected_fields=expected_fields,
                        expected_values=expected_values,
                        processing_time=processing_time,
                        field_scores={},
                        global_score=0.0,
                        error=str(e)
                    )
                    
                    print(f"      ❌ Error: {e}")
                
                results.append(result)
        
        # Cleanup model
        model.cleanup()
        
        # Print summary
        successful_results = [r for r in results if r.error is None]
        avg_score = sum(r.global_score for r in successful_results) / len(successful_results) if successful_results else 0.0
        
        print(f"\n📊 SUMMARY")
        print(f"   Total questions: {len(results)}")
        print(f"   Successful: {len(successful_results)}")
        print(f"   Average score: {avg_score:.3f}")
        print(f"   Average time: {sum(r.processing_time for r in results) / len(results):.2f}s")
        
        return results
    
    def save_responses(self, results: List[TestResult], output_dir: str = "responses"):
        """Save responses with path-based evaluation"""
        if not results:
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        model_name = results[0].model_name
        clean_name = re.sub(r'[^\w\-]', '_', model_name)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{clean_name}_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        # Calculate aggregate statistics
        successful_results = [r for r in results if r.error is None]
        avg_score = sum(r.global_score for r in successful_results) / len(successful_results) if successful_results else 0.0
        
        # Show detailed scores for each question
        print(f"\n📋 DETAILED QUESTION SCORES:")
        question_scores = {}
        for r in results:
            key = f"doc_{r.document_id}_{r.question_id}"
            question_scores[key] = {
                "question": r.question,
                "global_score": r.global_score,
                "field_scores": r.field_scores,
                "error": r.error,
                "processing_time": r.processing_time
            }
        
        # Print each question result
        for key, data in question_scores.items():
            status = "✅" if data["error"] is None else "❌"
            print(f"   {status} {key}: {data['global_score']:.3f}")
        
        # Show calculation
        successful_scores = [r.global_score for r in successful_results]
        print(f"   📊 AVERAGE CALCULATION:")
        all_scores = [r.global_score for r in results]
        print(f"   All scores (including failures): {[f'{score:.3f}' for score in all_scores]}")
        print(f"   Sum of scores: {sum(successful_scores):.3f}")
        print(f"   Number of successful responses: {len(successful_results)}")
        print(f"   Average score = {sum(successful_scores):.3f} / {len(successful_results)} = {avg_score:.3f}")
        
        # Save in compact format - only store extracted values, not full response
        data = {
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
            "total_questions": len(results),
            "successful": len(successful_results),
            "average_score": avg_score,
            "evaluation_method": "path_based",
            "responses": [
                {
                    "document_id": r.document_id,
                    "question_id": r.question_id,
                    "question": r.question,
                    "expected_fields": r.expected_fields,
                    "extracted_values": self._extract_field_values(r.predicted_answer, r.expected_fields) if not r.error else {},
                    "field_scores": r.field_scores,
                    "global_score": r.global_score,
                    "processing_time": r.processing_time,
                    "error": r.error
                }
                for r in results
            ]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Results saved to: {filepath}")
        return filepath
    
    def compare_responses(self, response_dir: str = "responses"):
        """Compare response files with path-based scores"""
        if not os.path.exists(response_dir):
            print("❌ No responses directory found")
            return
        
        files = [f for f in os.listdir(response_dir) if f.endswith('.json')]
        if len(files) < 2:
            print("❌ Need at least 2 response files for comparison")
            return
        
        print(f"\n📊 MODEL COMPARISON")
        print("=" * 50)
        
        models_data = []
        
        for filename in sorted(files):
            filepath = os.path.join(response_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                model_name = data.get('model_name', 'Unknown')
                successful = data.get('successful', 0)
                total = data.get('total_questions', 0)
                avg_score = data.get('average_score', 0.0)
                
                models_data.append({
                    'name': model_name,
                    'file': filename,
                    'success_rate': successful / total if total > 0 else 0,
                    'avg_score': avg_score,
                    'successful': successful,
                    'total': total
                })
                
            except Exception as e:
                print(f"⚠️  Could not load {filename}: {e}")
        
        # Sort by average score
        models_data.sort(key=lambda x: x['avg_score'], reverse=True)
        
        print(f"{'Rank':<4} {'Model':<30} {'Score':<8} {'Success':<10} {'File'}")
        print("-" * 70)
        
        for i, model in enumerate(models_data, 1):
            print(f"{i:<4} {model['name'][:29]:<30} {model['avg_score']:.3f}   {model['successful']}/{model['total']:<6} {model['file']}")

def main():
    parser = argparse.ArgumentParser(description="Path-Based AI Model Tester")
    parser.add_argument("--model", type=str, help="Model specification (e.g., ollama:qwen3:8b)")
    parser.add_argument("--category", type=str, help="Filter by category")
    parser.add_argument("--quick", action="store_true", help="Quick test (2 documents)")
    parser.add_argument("--compare-responses", action="store_true", help="Compare saved responses")
    parser.add_argument("--dataset", type=str, default="dataset.json", help="Dataset file")
    parser.add_argument("--output-dir", type=str, default="responses", help="Output directory")
    
    args = parser.parse_args()
    
    tester = PathBasedModelTester(args.dataset)
    
    if args.compare_responses:
        tester.compare_responses(args.output_dir)
    
    elif args.model:
        max_docs = 2 if args.quick else None
        
        results = tester.test_model(
            model_spec=args.model,
            category_filter=args.category,
            max_documents=max_docs
        )
        
        if results:
            tester.save_responses(results, args.output_dir)
    
    else:
        print("❌ Please specify --model or --compare-responses")
        print("\nExample usage:")
        print("  python model_evaluator.py --model ollama:qwen3:8b --quick")
        print("  python model_evaluator.py --model transformers:microsoft/Phi-3.5-mini-instruct")
        print("  python model_evaluator.py --compare-responses")

if __name__ == "__main__":
    main()