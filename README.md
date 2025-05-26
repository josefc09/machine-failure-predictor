# Machine Failure Predictor
Machine Learning solution that predicts when a machine will fail using  Microsoft Azure Predictive Maintenance dataset.

---


## Project structure

```bash
.
├── app
│   ├── endpoints.py
│   ├── example_api.json
│   ├── inference.py
│   ├── __init__.py
│   ├── model_loader.py
│   └── schemas.py
├── config.yaml
├── data
│   ├── synthetic_data
│   └── training_data
├── docs
│   ├── Short technical report.md
│   └── Technical desing.md
├── LICENSE
├── logs
├── main.py
├── models
│   └── model.joblib
├── outputs
├── README.md
├── requirements.txt
├── src
│   ├── data.py
│   ├── evaluate.py
│   ├── model.py
│   ├── predict.py
│   └── train.py
```

## Documentation

Documentation

This project includes specific design and reporting documents located in the docs/ folder:

* `docs/Short technical report.md`: Provides a concise summary of the project, methodology, and results.
* `docs/Technical design.md`: Details the technical architecture, data flow, model design, and API implementation.

Refer to these documents for in-depth information about the project's design and findings.

## Getting Started

Follow these instructions to set up and run the project locally.

### Prerequisites

* Python 3.12.3
* Git

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**
    * **Linux / macOS:**
        ```bash
        source venv/bin/activate
        ```
    * **Windows:**
        ```bash
        .\venv\Scripts\activate
        ```

4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Data Setup:**
    * Download the [Microsoft Azure Predictive Maintenance dataset](https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance).
    * Place the `PdM_telemetry.csv`, `PdM_errors.csv`, and `PdM_failures.csv` files into the appropriate directories as specified in your `config.yaml` (e.g., `data/training_data/`). You might need to create these directories.

6.  **Configuration:**
    * Review and adjust `config.yaml` to match your directory structure and desired parameters.

---

## Usage

Ensure your virtual environment is active before running any commands.

### Training the Model

This command runs the training script using the data and parameters defined in `config.yaml`. It will preprocess the data, train the XGBoost model, evaluate it, and save the model artifact (e.g., `model.joblib`) and evaluation results.

```bash
python src/train.py
```

Running Batch Predictions

This command uses the trained model to make predictions on synthetic data (as specified by synthetic_data_folder in config.yaml). It loads the model, processes the new data, and outputs the risk predictions.

```bash
python src/predict.py
```

Running the RESTful API Service

This command starts the FastAPI server. The `--reload` flag enables auto-reloading during development. Not recommended for production.

```bash
uvicorn main:app --host localhost --port 8000 --reload
```

* `main:app`: Assumes your FastAPI application instance (app) is in a file named main.py. Adjust if necessary.
* `--host localhost`: Makes the server accessible on your network. Use 127.0.0.1 to restrict it to your local machine.
* `--port 8000`: Specifies the port to run on.


### Testing the API

Once the server is running, you can interact with the API:

1.  **Interactive Documentation (Swagger UI):**

    Open your browser and navigate to:

    [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

    This provides an interactive interface where you can test the `/predict` endpoint directly.

2.  **Using `curl` (Example):**

    You can send a POST request to the `/predict` endpoint using `curl` or a tool like Postman. You'll need to provide data in the expected JSON format.

    ```bash
    curl -X 'POST' \
      '[http://127.0.0.1:8000/predict](http://127.0.0.1:8000/predict)' \
      -H 'accept: application/json' \
      -H 'Content-Type: application/json' \
      -d '[
      {
        "machineID": 1,
        "telemetryLast24h": [
          {"datetime": "2025-05-24T06:00:00", "machineID": 1, "volt": 176.2, "rotate": 418.5, "pressure": 113.0, "vibration": 45.0},
          {"datetime": "2025-05-24T07:00:00", "machineID": 1, "volt": 170.1, "rotate": 420.0, "pressure": 110.0, "vibration": 46.1}
        ],
        "errorsLast24h": [
          {"datetime": "2025-05-24T06:30:00", "machineID": 1, "errorID": "error1"}
        ]
      }
    ]'
    ```