import pandas as pd

def load_and_merge_data(telemetry_path, errors_path, failures_path=None):
    """Loads and merges telemetry, errors, and optionally failures data."""
    df_telemetry = pd.read_csv(telemetry_path, parse_dates=['datetime'])
    df_errors = pd.read_csv(errors_path, parse_dates=['datetime'])

    if failures_path:
        df_failures = pd.read_csv(failures_path, parse_dates=['datetime'])
        return df_telemetry, df_errors, df_failures
    else:
        return df_telemetry, df_errors, None

def create_hourly_error_counts(df_errors):
    """Creates hourly error counts per machine."""
    hourly_error_counts = df_errors.groupby(['machineID', pd.Grouper(key='datetime', freq='h')]).size().reset_index(name='countErrors')
    return hourly_error_counts

def preprocess_data(df_telemetry, hourly_error_counts, df_failures=None, is_train=True):
    """Preprocesses data for training and prediction."""
    df_final = pd.merge(df_telemetry, hourly_error_counts, on=['machineID', 'datetime'], how='left')
    df_final['countErrors'] = df_final['countErrors'].fillna(0)
    df_final = df_final.sort_values(by=['machineID', 'datetime'])
    df_final.set_index('datetime', inplace=True)

    telemetry_cols = ['volt', 'rotate', 'pressure', 'vibration']
    for col in telemetry_cols:
        rolling_features = df_final.groupby('machineID')[col].rolling(window='24h', min_periods=1)
        df_final[f'{col}_24h_mean'] = rolling_features.mean().reset_index(level=0, drop=True)
        df_final[f'{col}_24h_std'] = rolling_features.std().reset_index(level=0, drop=True)

    df_final['errors_in_24h'] = df_final.groupby('machineID')['countErrors'].rolling(window='24h', min_periods=1).sum().reset_index(level=0, drop=True)
    df_final.reset_index(inplace=True)
    df_final.sort_values(by=['datetime', 'machineID'], inplace=True)
    df_final = df_final.drop(columns=['countErrors']) # Drop base count after rolling sum
    df_final = df_final.fillna(0)


    if is_train:
        df_next_failure = df_failures[['machineID', 'datetime']].rename(columns={'datetime': 'next_failure_datetime'})
        df_next_failure.sort_values(by=['next_failure_datetime', 'machineID'], inplace=True)
        df_final = pd.merge_asof(
            left=df_final,
            right=df_next_failure,
            left_on='datetime',
            right_on='next_failure_datetime',
            by='machineID',
            direction='forward'
        )
        time_to_failure = df_final['next_failure_datetime'] - df_final['datetime']
        df_final['failure_in_next_24h'] = ((time_to_failure.dt.total_seconds() <= (24 * 3600)) & (time_to_failure.dt.total_seconds() > 0)).astype(int)
        df_final = df_final.drop(columns=['next_failure_datetime'])
        df_final = df_final.fillna(0) # Fill NaNs again after merge
        return df_final
    else:
        return df_final

def prepare_data_for_training(df_final):
    """Prepares data for model training."""
    features = [col for col in df_final.columns if col not in ['datetime', 'machineID', 'failure_in_next_24h']]
    X = df_final[features]
    y = df_final['failure_in_next_24h']
    return X, y, features

def split_data(X, y, df_final, train_size=0.8):
    """Splits data into training and testing sets chronologically."""
    # We need machineID for evaluation, but not for training
    # Keep original X and y, but split based on df_final index
    split_point = int(len(df_final) * train_size)
    
    X_train = X.iloc[:split_point]
    X_test_full = df_final.iloc[split_point:].copy() # Keep df_final structure for test
    y_train = y.iloc[:split_point]
    y_test = y.iloc[split_point:]
    
    # Return X_test_full which includes machineID for evaluation purposes
    return X_train, X_test_full, y_train, y_test