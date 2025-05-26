import pandas as pd
from datetime import datetime
from typing import List, Dict, Any
from src.data import create_hourly_error_counts, preprocess_data
from app.model_loader import get_model, get_model_features
from app.schemas import MachineDataInput

def prepare_dataframe_from_json(machine_data: MachineDataInput) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Converts telemetry and error data from the JSON input for a single machine
    into Pandas DataFrames.
    """
    telemetry_list = [record.dict() for record in machine_data.telemetryLast24h]
    errors_list = [record.dict() for record in machine_data.errorsLast24h]

    df_telemetry = pd.DataFrame(telemetry_list)
    df_errors = pd.DataFrame(errors_list)

    if not df_telemetry.empty:
        df_telemetry['datetime'] = pd.to_datetime(df_telemetry['datetime'])
    if not df_errors.empty:
        # If there are no errors, create an empty DataFrame with the expected columns
        df_errors['datetime'] = pd.to_datetime(df_errors['datetime'])
    else: # Handle the case where there are no errors
         df_errors = pd.DataFrame(columns=['datetime', 'machineID', 'errorID'])
         df_errors['datetime'] = pd.to_datetime(df_errors['datetime'])

    return df_telemetry, df_errors


def run_inference_for_machine(machine_data_input: MachineDataInput) -> Dict[str, Any]:
    """
    Runs preprocessing and inference for a single machine's data.
    """
    model = get_model()
    model_features = get_model_features()

    df_telemetry, df_errors = prepare_dataframe_from_json(machine_data_input)

    if df_telemetry.empty:
        return {
            "machineId": machine_data_input.machineId,
            "predictionDate": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "riskOfFailure": "N/A (No telemetry data)"
        }

    if not df_errors.empty:
        hourly_error_counts = create_hourly_error_counts(df_errors)
    else: 
        hourly_error_counts = pd.DataFrame(columns=['machineID', 'datetime', 'countErrors'])

    df_processed = preprocess_data(df_telemetry, hourly_error_counts, is_train=False)

    if df_processed.empty:
        return {
            "machineId": machine_data_input.machineId,
            "predictionDate": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "riskOfFailure": "N/A (Preprocessing failed or no data)"
        }

    latest_data_point = df_processed.sort_values(by='datetime', ascending=False).iloc[[0]]

    X_predict = pd.DataFrame(columns=model_features)
    for col in model_features:
        if col in latest_data_point.columns:
            X_predict[col] = latest_data_point[col]
        else:
            X_predict[col] = 0 

    X_predict = X_predict[model_features]

    probability_failure = model.predict_proba(X_predict)[:, 1][0] # Probability of class '1' (failure)
    risk_percentage = f"{probability_failure * 100:.1f}%"

    return {
        "machineId": machine_data_input.machineId,
        "predictionDate": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "riskOfFailure": risk_percentage
    }

def batch_predict(batch_input: List[MachineDataInput]) -> List[Dict[str, Any]]:
    """
    Processes a batch of machine data inputs and returns predictions.
    """
    results = []
    for machine_data in batch_input:
        try:
            prediction = run_inference_for_machine(machine_data)
            results.append(prediction)
        except Exception as e:
            print(f"Error processing machine {machine_data.machineId}: {e}")
            results.append({
                "machineId": machine_data.machineId,
                "predictionDate": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "riskOfFailure": f"Error: {e}"
            })
    return results