from transformers import TFT5ForConditionalGeneration, AutoTokenizer
import logging

logger = logging.getLogger(__name__)

# Model are kept here for global use
ml_models = {}

def load_model(model_path):
    model = TFT5ForConditionalGeneration.from_pretrained(model_path)
    return model

def load_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return tokenizer

async def load_models_on_startup():
    try:
        model_path = "./pretrained/finance_chatbot_0615_1817"
        logger.info(f"Loading fine-tuned model from: {model_path}")
        
        model = load_model(model_path)
        tokenizer = load_tokenizer(model_path)
        
        ml_models["model"] = model
        ml_models["tokenizer"] = tokenizer
        
        logger.info("Fine-tuned model loaded successfully!")
        
    except Exception as e:
        logger.error(f"Failed to load fine-tuned model: {e}")
        raise e

async def cleanup_models():
    logger.info("Cleaning up resources...")
    ml_models.clear()

def get_models():
    if "model" not in ml_models or "tokenizer" not in ml_models:
        raise ValueError("Model not loaded")
    return ml_models["model"], ml_models["tokenizer"]