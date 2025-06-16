from typing import Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

logger = logging.getLogger(__name__)

class ModelHandler:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def initialize(self, model_id: str) -> None:
        """
        Initialize the model and tokenizer.
        
        Args:
            model_id (str): The Hugging Face model ID to load
        """
        try:
            logger.info(f"Loading model {model_id} on {self.device}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(model_id)
            self.model.to(self.device)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
            
    def preprocess(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess the input data.
        
        Args:
            input_data (Dict[str, Any]): The input data containing the question
            
        Returns:
            Dict[str, Any]: Preprocessed input data
        """
        if not isinstance(input_data, dict):
            raise ValueError("Input data must be a dictionary")
            
        if "question" not in input_data:
            raise ValueError("Input data must contain 'question' field")
            
        return input_data
        
    def inference(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform inference on the input data.
        
        Args:
            input_data (Dict[str, Any]): The preprocessed input data
            
        Returns:
            Dict[str, Any]: The model's prediction
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not initialized. Call initialize() first.")
            
        try:
            question = input_data["question"]
            max_length = input_data.get("max_length", 100)
            
            # Tokenize input
            inputs = self.tokenizer(question, return_tensors="pt").to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Calculate confidence based on response length
            confidence = "high" if len(answer.split()) > 20 else "medium" if len(answer.split()) > 10 else "low"
            
            return {
                "question": question,
                "answer": answer,
                "confidence": confidence
            }
            
        except Exception as e:
            logger.error(f"Inference failed: {str(e)}")
            raise
            
    def handle(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main handler method that processes the input data through the pipeline.
        
        Args:
            input_data (Dict[str, Any]): The input data
            
        Returns:
            Dict[str, Any]: The model's prediction
        """
        try:
            preprocessed_data = self.preprocess(input_data)
            result = self.inference(preprocessed_data)
            return result
        except Exception as e:
            logger.error(f"Handler failed: {str(e)}")
            raise 