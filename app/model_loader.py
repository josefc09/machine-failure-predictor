import joblib
import yaml
import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent 

# Global variables to cache the loaded model, config, and its features.
MODEL = None
CONFIG = None
MODEL_FEATURES = None

def load_config(config_path=BASE_DIR / "config.yaml"):
    """Loads the YAML configuration file into the global CONFIG variable."""
    global CONFIG
    # Load only if it hasn't been loaded yet (singleton pattern).
    if CONFIG is None:
        try:
            with open(config_path, 'r') as f:
                CONFIG = yaml.safe_load(f)
            logger.info(f"Configuration loaded successfully from {config_path}")
        except FileNotFoundError:
            logger.error(f"Configuration file not found at: {config_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading configuration file: {e}")
            raise
    return CONFIG

def load_prediction_model():
    """Loads the trained prediction model and its features into global variables."""
    global MODEL, CONFIG, MODEL_FEATURES
    # Load only if the model hasn't been loaded yet.
    if MODEL is None:
        # Ensure config is loaded first to get the model path.
        if CONFIG is None:
            load_config()
        
        # Construct the full path to the model file.
        model_path_str = CONFIG['paths']['model_output']
        model_path = BASE_DIR / model_path_str
        
        try:
            MODEL = joblib.load(model_path)
            # Store the feature names expected by the model for later validation/use.
            MODEL_FEATURES = MODEL.feature_names_in_.tolist() 
            logger.info(f"Model loaded successfully from {model_path}")
        except FileNotFoundError:
            logger.error(f"Model file not found at: {model_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    return MODEL, MODEL_FEATURES

# --- Eager Loading ---
# Load configuration and model when this module is first imported.
# This prevents loading delays on the first API request (e.g., in a FastAPI startup event).
load_config()
load_prediction_model()

# --- Accessor Functions ---
# Provide a controlled way to access the global variables,
# ensuring they are loaded if accessed before initial loading (though unlikely with eager loading).

def get_model():
    """Returns the loaded model."""
    if MODEL is None:
        load_prediction_model()
    return MODEL

def get_model_features():
    """Returns the list of feature names the model expects."""
    if MODEL_FEATURES is None:
        load_prediction_model()
    return MODEL_FEATURES

def get_config():
    """Returns the loaded configuration."""
    if CONFIG is None:
        load_config()
    return CONFIG