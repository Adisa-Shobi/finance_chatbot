from pydantic import BaseModel

# Pydantic Models
class QuestionRequest(BaseModel):
    question: str
    max_length: int = 200
    temperature: float = 0.7

class AnswerResponse(BaseModel):
    question: str
    answer: str
    confidence: str