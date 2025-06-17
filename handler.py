from transformers import TFT5ForConditionalGeneration, AutoTokenizer

class EndpointHandler:
    def __init__(self, path=""):
        """Load model and tokenizer"""
        self.model = TFT5ForConditionalGeneration.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
    
    def __call__(self, data):
        """Generate response"""
        # Get input
        question = data.get("inputs", "")
        
        # Format input
        input_text = f"Answer this financial question based on Warren Buffett's principles: {question}"
        
        # Tokenize
        inputs = self.tokenizer(input_text, return_tensors="tf", max_length=256, truncation=True)
        
        # Generate
        outputs = self.model.generate(
            inputs["input_ids"],
            max_length=200,
            num_beams=4,
            early_stopping=True
        )
        
        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return {"generated_text": response}
