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

# Get allowed origins from environment variable
allowed_origins = os.environ.get("ALLOWED_ORIGINS", "*").split(",")
logger.info(f"Allowed CORS origins: {allowed_origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


if __name__ == "__main__":
    uvicorn.run(
        app=app
    )