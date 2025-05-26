import pandas as pd
from data import load_and_merge_data, create_hourly_error_counts, preprocess_data
from model import load_model
import os
import argparse
import yaml
import logging

def setup_logging(log_path):
    """Configura el logging básico."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, mode='a'), # append mode
            logging.StreamHandler()
        ]
    )

def load_config(config_path='config.yaml'):
    """ Loading YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logging.error(f"Configuration file not found at: {config_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading configuration file: {e}")
        raise

def predict_latest_failures(config):
    """Performs failure predictions using new data from config paths."""
    
    paths = config['paths']
    new_data_folder = paths['new_data_folder']
    model_path = paths['model_output']
    output_path = paths['predictions_output']

    telemetry_path = os.path.join(new_data_folder, 'PdM_telemetry.csv')
    errors_path = os.path.join(new_data_folder, 'PdM_errors.csv')

    logging.info("--- Starting Prediction Process ---")

    if not os.path.exists(telemetry_path):
        logging.error(f"Cannot find 'PdM_telemetry.csv' in {new_data_folder}")
        return
    if not os.path.exists(errors_path):
        logging.error(f"Cannot find 'PdM_errors.csv' in {new_data_folder}")
        return

    try:
        model = load_model(model_path)
        model_features = model.feature_names_in_.tolist()

        logging.info(f"Loading new data from {new_data_folder}...")
        nuevos_df_telemetry, nuevos_df_errors, _ = load_and_merge_data(telemetry_path, errors_path)
        
        logging.info("Preprocessing new data...")
        nuevos_hourly_error_counts = create_hourly_error_counts(nuevos_df_errors)
        nuevos_df_final = preprocess_data(nuevos_df_telemetry, nuevos_hourly_error_counts, is_train=False)

        if nuevos_df_final.empty:
            logging.warning("No data after preprocessing. Cannot predict.")
            return

        logging.info("Filtering for latest data per machine...")
        last_dates = nuevos_df_final.groupby('machineID')['datetime'].max().reset_index()
        latest_data = pd.merge(nuevos_df_final, last_dates, on=['machineID', 'datetime'], how='inner')

        if latest_data.empty:
            logging.warning("No latest data found. Cannot predict.")
            return

        logging.info("Aligning features and predicting...")
        current_features = [col for col in latest_data.columns if col in model_features]
        missing_cols = set(model_features) - set(current_features)
        for col in missing_cols:
            latest_data[col] = 0 
        
        X_nuevos_latest = latest_data[model_features]
        nuevas_probabilidades_decimal = model.predict_proba(X_nuevos_latest)[:, 1]
        nuevas_probabilidades_porcentaje = nuevas_probabilidades_decimal * 100
        nuevas_probabilidades_formateadas = [f"{prob:.1f}%" for prob in nuevas_probabilidades_porcentaje]

        resultados_prediccion = pd.DataFrame({
            'machineID': latest_data['machineID'],
            'datetime': latest_data['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S'),
            'failure risk (24h)': nuevas_probabilidades_formateadas
        })

        print("\nFailure Prediction Results (Based on the last entry per machine):")
        print(resultados_prediccion)

        logging.info(f"Saving predictions to {output_path}...")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        resultados_prediccion.to_csv(output_path, index=False)
        logging.info(f"Predictions saved successfully.")
        logging.info("--- Prediction Process Finished Successfully ---")

    except Exception as e:
        logging.error(f"An error occurred during prediction: {e}", exc_info=True)
        logging.info("--- Prediction Process Failed ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Perform failure predictions on new data.")
    parser.add_argument("--config", type=str, default='config.yaml', help="Path to the configuration file.")
    args = parser.parse_args()

    # Load config and configure logging
    config_data = load_config(args.config)
    setup_logging(config_data['paths']['log_file'])
    
    # Run prediction
    predict_latest_failures(config_data)