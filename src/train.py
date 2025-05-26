import pandas as pd
from data import load_and_merge_data, create_hourly_error_counts, preprocess_data, prepare_data_for_training, split_data
from model import train_model, plot_feature_importance, save_model 
from evaluate import evaluate_and_save, evaluate_on_specific_machines
import argparse
import os
import yaml 
import logging 

def setup_logging(log_path):
    """Configure logging."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )

def load_config(config_path='config.yaml'):
    """Loading YAML configuration."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logging.error(f"Configuration file not found at: {config_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading configuration file: {e}")
        raise

if __name__ == "__main__":
    config = load_config()
    setup_logging(config['paths']['log_file'])

    logging.info("--- Starting Training Process ---")

    try:
        paths = config['paths']
        train_params = config['training']
        model_cfg = config['model_params']

        logging.info("Loading and Preprocessing data...")
        df_telemetry, df_errors, df_failures = load_and_merge_data(paths['training_telemetry'], paths['training_errors'], paths['training_failures'])
        hourly_error_counts = create_hourly_error_counts(df_errors)
        df_final = preprocess_data(df_telemetry, hourly_error_counts, df_failures=df_failures, is_train=True)
        X, y, features = prepare_data_for_training(df_final)
        X_train, X_test, y_train, y_test = split_data(X, y, df_final, train_size=train_params['train_size'])

        # --- Calcular scale_pos_weight ---
        scale_pos_weight_value = 1
        if 1 in y_train.value_counts() and y_train.value_counts()[1] > 0:
            neg_cases = y_train.value_counts()[0]
            pos_cases = y_train.value_counts()[1]
            scale_pos_weight_value = neg_cases / pos_cases
            logging.info(f"Calculated Scale Pos Weight: {scale_pos_weight_value:.2f}")
        else:
            logging.warning("No positive cases in training data. Using scale_pos_weight = 1.")

        # --- Training the model ---
        X_train_features = X_train[features]
        model = train_model(X_train_features, y_train, model_cfg, scale_pos_weight=scale_pos_weight_value)

        # --- Model evaluation ---
        evaluate_and_save(
            model, 
            X_test,
            y_test, 
            paths['evaluation_metrics'], 
            paths['confusion_matrix_plot']
        )
        evaluate_on_specific_machines(model, X_test, y_test)

        plot_feature_importance(model, filepath=paths['feature_importance_plot'])
        save_model(model, filepath=paths['model_output'])

        logging.info("--- Training Process Finished Successfully ---")

    except Exception as e:
        logging.error(f"An error occurred during training: {e}", exc_info=True)
        logging.info("--- Training Process Failed ---")