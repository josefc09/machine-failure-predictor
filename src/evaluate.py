from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import logging

def plot_and_save_confusion_matrix(y_test, y_pred, filepath):
    """Generates, displays, and saves the confusion matrix."""
    try:
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        
        sns.heatmap(cm, 
                    annot=True,     
                    fmt='d',        
                    cmap='Blues',  
                    linewidths=.5, 
                    linecolor='lightgray',
                    xticklabels=['No Failure', 'Failure'],
                    yticklabels=['No Failure', 'Failure'])

        plt.title('Confusion Matrix', fontsize=16)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10, rotation=0)
        plt.tight_layout()

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        plt.savefig(filepath)
        logging.info(f"Confusion matrix saved to: {filepath}")
        plt.close()

    except Exception as e:
        logging.error(f"Error saving the confusion matrix: {e}")

def save_metrics(y_test, y_pred, filepath):
    """Calculates and saves evaluation metrics to a JSON file."""
    try:
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        metrics_data = {
            'accuracy': accuracy,
            'classification_report': report
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(metrics_data, f, indent=4)
        logging.info(f"Evaluation metrics saved to: {filepath}")

    except Exception as e:
        logging.error(f"Error saving metrics: {e}")


def evaluate_and_save(model, X_test, y_test, metrics_path, cm_plot_path):
    """Performs the full evaluation and saves the results."""
    logging.info("Performing full model evaluation...")
    
    features = model.feature_names_in_
    X_test_features = X_test[features] 
    y_pred = model.predict(X_test_features)

    print("\nOverall evaluation on the test set:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))

    # Save metrics and matrix
    save_metrics(y_test, y_pred, metrics_path)
    plot_and_save_confusion_matrix(y_test, y_pred, cm_plot_path)
    
    logging.info("Full evaluation and saving completed.")
    return y_pred


def evaluate_on_specific_machines(model, X_test, y_test, num_machines=2):
    """Evaluates the model."""

    logging.info(f"\nPerforming evaluation on {num_machines} specific machines...")
    test_machines = X_test['machineID'].unique()
    machines_to_test = test_machines[:num_machines]

    for machine_id in machines_to_test:
        machine_mask = X_test['machineID'] == machine_id
        X_test_machine = X_test[machine_mask]
        y_test_machine = y_test[machine_mask]

        if not X_test_machine.empty:
            X_test_machine_features = X_test_machine[model.feature_names_in_]
            y_pred_machine = model.predict(X_test_machine_features)
            
            print(f"\nResults for machine {machine_id}:")
            print("Accuracy:", accuracy_score(y_test_machine, y_pred_machine))
            print("Classification Report:\n", classification_report(y_test_machine, y_pred_machine, zero_division=0))
        else:
            print(f"\nNo data found for machine {machine_id} in the test set.")