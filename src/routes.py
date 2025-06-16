from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import logging
from src.predict import predict_answer
from src.loaders import get_models
from src.models import AnswerResponse, QuestionRequest

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

@router.post("/predict", response_model=AnswerResponse)
async def predict(request: QuestionRequest):
    try:
        model, tokenizer = get_models()
    except ValueError as e:
        logger.error("Model not loaded when prediction requested")
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        logger.info(f"Generating prediction for question: {request.question[:50]}...")
        
        answer = predict_answer(
            question=request.question,
            model=model,
            tokenizer=tokenizer,
            max_length=request.max_length
        )
        
        confidence = "high" if len(answer.split()) > 20 else "medium" if len(answer.split()) > 10 else "low"
        
        logger.info(f"Prediction generated successfully with {confidence} confidence")
        
        return AnswerResponse(
            question=request.question,
            answer=answer,
            confidence=confidence
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.get("/")
async def root():
    return {"message": "Warren Buffett Financial Advisor API (Fine-tuned)", "status": "running"}

@router.get("/health")
async def health():
    try:
        get_models()
        return {"status": "healthy", "message": "Model ready"}
    except ValueError:
        return {"status": "error", "message": "Model not loaded"}
