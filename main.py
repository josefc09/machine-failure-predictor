import sys
from pathlib import Path
import logging
import uvicorn
from fastapi import FastAPI
from app.endpoints import router as prediction_router
from app.model_loader import load_prediction_model, load_config

ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))

app = FastAPI(
    title="Predictive Maintenance API",
    description="API for predicting machine failures based on telemetry and error data.",
    version="1.0.0",
    openapi_tags=[
        {
            "name": "Predictions",
            "description": "Endpoints for making failure predictions.",
        }
    ]
)

# --- Startup Event Handler ---
@app.on_event("startup")
async def startup_event():
    """
    Loads the configuration and the ML model when the application starts.
    This ensures they are ready before the first request arrives.
    """
    logging.info("Application startup: Loading model and config...")
    load_config() # Ensures config is loaded.
    load_prediction_model() # Loads the ML model.
    logging.info("Model and config loaded successfully.")

app.include_router(prediction_router, prefix="/api/v1")

# --- Root Endpoint ---
@app.get("/", tags=["Root"])
async def read_root():
    """
    Provides a simple welcome message and a link to the API documentation.
    """
    return {"message": "Welcome to the Predictive Maintenance API. Go to /docs for API documentation."}

# --- Main Execution Block ---
if __name__ == "__main__":
    """
        Note: For development, add '--reload'. For production, remove '--reload'.
    """
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True) 