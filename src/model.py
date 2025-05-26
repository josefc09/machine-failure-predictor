from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import matplotlib.pyplot as plt
import joblib
import os
import logging


def train_model(X_train, y_train, model_params, scale_pos_weight=None):
    """Trains the XGBoost model using parameters from config."""
    logging.info("Training XGBoost model...")
    model = XGBClassifier(
        eval_metric=model_params.get('eval_metric', 'logloss'),
        random_state=model_params.get('random_state', 42),
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False
    )
    model.fit(X_train, y_train)
    logging.info("Model training complete.")
    return model

def evaluate_training_model(model, X_test, y_test):
    """Evaluates the model on the test set during training."""
    logging.info("Evaluating model on the test set...")
    features = model.feature_names_in_
    X_test_features = X_test[features]
    y_pred = model.predict(X_test_features)
    
    logging.debug("\nConfusion Matrix:\n" + str(confusion_matrix(y_test, y_pred)))
    logging.info("\nClassification Report:\n" + classification_report(y_test, y_pred))

def plot_feature_importance(model, filepath):
    """Plots and saves feature importance to a file."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        logging.info(f"Generating and saving feature importance plot to: {filepath}")
        fig, ax = plt.subplots(figsize=(10, 8))
        xgb.plot_importance(model, ax=ax)
        plt.title('Feature Importance (F-score)', fontsize=16)
        plt.tight_layout()
        plt.savefig(filepath)
        logging.info(f"Feature importance plot saved successfully.")
        plt.close(fig)

    except Exception as e:
        logging.error(f"Error saving feature importance plot: {e}")

def save_model(model, filepath): 
    """Saves the trained model to the specified path."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(model, filepath)
        logging.info(f"Model saved successfully at: {filepath}")
    except Exception as e:
        logging.error(f"Failed to save model at {filepath}. Error: {e}")
        raise

def load_model(filepath):
    """Loads a previously trained model from the specified path."""
    try:
        logging.info(f"Loading model from: {filepath}")
        model = joblib.load(filepath)
        logging.info("Model loaded successfully.")
        return model
    except FileNotFoundError:
        logging.error(f"Model file not found at: {filepath}")
        raise
    except Exception as e:
        logging.error(f"Failed to load model from {filepath}. Error: {e}")
        raise