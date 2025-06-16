#!/usr/bin/env python
from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from src.loaders import *
from dotenv import load_dotenv
import logging
import uvicorn
import os
from src.routes import router
from src.loaders import load_models_on_startup, cleanup_models

# Load environment
load_dotenv()

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models on startup
    await load_models_on_startup()
    yield
    # Clean up on shutdown
    await cleanup_models()

app = FastAPI(
    title="Financial Advisor Chatbot (Warren Buffet)",
    description="API for making predictions using a finetuned t5 model",
    version="1.0.0",
    lifespan=lifespan
)

# Get CORS settings from environment
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
CORS_ALLOW_CREDENTIALS = os.getenv("CORS_ALLOW_CREDENTIALS", "true").lower() == "true"
CORS_ALLOW_METHODS = os.getenv("CORS_ALLOW_METHODS", "GET,POST,PUT,DELETE,OPTIONS").split(",")
CORS_ALLOW_HEADERS = os.getenv("CORS_ALLOW_HEADERS", "*").split(",")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=CORS_ALLOW_CREDENTIALS,
    allow_methods=CORS_ALLOW_METHODS,
    allow_headers=CORS_ALLOW_HEADERS,
)

logger.info(f"CORS enabled for origins: {ALLOWED_ORIGINS}")

app.include_router(router)


if __name__ == "__main__":
    uvicorn.run(
        app=app
    )