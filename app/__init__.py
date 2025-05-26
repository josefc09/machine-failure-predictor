# app/__init__.py
# This file handles the initialization for the 'app' package.
# Its main purpose here is to set up application-wide logging.

import logging
import sys
from app.model_loader import load_config # Import the configuration loading function.

# Load configuration on import to get paths and other parameters.
config = load_config() 
log_file_path = config.get('paths', {}).get('log_file', 'logs/fastapi_app.log')

# Configure the logging system.
logging.basicConfig(
    level=logging.INFO, # Set the minimum logging level.
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", # Define the format for log messages.
    handlers=[
        logging.FileHandler(log_file_path, mode='a'), # Log to a file (append mode).
        logging.StreamHandler(sys.stdout) # Also log to the standard console.
    ]
)

# Get a logger for this module and log the initialization.
logger = logging.getLogger(__name__)
logger.info("Application 'app' package initialized and logging configured.")