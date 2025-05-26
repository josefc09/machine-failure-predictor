from fastapi import APIRouter, HTTPException, Body
from typing import List
from app.schemas import PredictionInput, PredictionResponse, MachineDataInput, PredictionOutputRecord
from app.inference import batch_predict
import logging

# Setup router and logger for this module
router = APIRouter()
logger = logging.getLogger(__name__)

api_examples = {
    "single_machine": {
        "summary": "A single machine example",
        "description": "Predict failure for one machine.",
        "value": [
              {
    "machineID": 1,
    "telemetryLast24h": [
      {
        "datetime": "2025-05-23T00:00:00",
        "machineID": 1,
        "volt": 160.0,
        "rotate": 1080.0,
        "pressure": 100.0,
        "vibration": 48.0
      },
      {
        "datetime": "2025-05-23T01:00:00",
        "machineID": 1,
        "volt": 159.0,
        "rotate": 1076.0,
        "pressure": 99.6,
        "vibration": 48.4
      },
      {
        "datetime": "2025-05-23T02:00:00",
        "machineID": 1,
        "volt": 158.0,
        "rotate": 1072.0,
        "pressure": 99.2,
        "vibration": 48.8
      },
      {
        "datetime": "2025-05-23T03:00:00",
        "machineID": 1,
        "volt": 157.0,
        "rotate": 1068.0,
        "pressure": 98.8,
        "vibration": 49.2
      },
      {
        "datetime": "2025-05-23T04:00:00",
        "machineID": 1,
        "volt": 156.0,
        "rotate": 1064.0,
        "pressure": 98.4,
        "vibration": 49.6
      },
      {
        "datetime": "2025-05-23T05:00:00",
        "machineID": 1,
        "volt": 155.0,
        "rotate": 1060.0,
        "pressure": 98.0,
        "vibration": 50.0
      },
      {
        "datetime": "2025-05-23T06:00:00",
        "machineID": 1,
        "volt": 154.0,
        "rotate": 1056.0,
        "pressure": 97.6,
        "vibration": 50.4
      },
      {
        "datetime": "2025-05-23T07:00:00",
        "machineID": 1,
        "volt": 153.0,
        "rotate": 1052.0,
        "pressure": 97.2,
        "vibration": 50.8
      },
      {
        "datetime": "2025-05-23T08:00:00",
        "machineID": 1,
        "volt": 152.0,
        "rotate": 1048.0,
        "pressure": 96.8,
        "vibration": 51.2
      },
      {
        "datetime": "2025-05-23T09:00:00",
        "machineID": 1,
        "volt": 151.0,
        "rotate": 1044.0,
        "pressure": 96.4,
        "vibration": 51.6
      },
      {
        "datetime": "2025-05-23T10:00:00",
        "machineID": 1,
        "volt": 150.0,
        "rotate": 1040.0,
        "pressure": 96.0,
        "vibration": 52.0
      },
      {
        "datetime": "2025-05-23T11:00:00",
        "machineID": 1,
        "volt": 149.0,
        "rotate": 1036.0,
        "pressure": 95.6,
        "vibration": 52.4
      },
      {
        "datetime": "2025-05-23T12:00:00",
        "machineID": 1,
        "volt": 148.0,
        "rotate": 1032.0,
        "pressure": 95.2,
        "vibration": 52.8
      },
      {
        "datetime": "2025-05-23T13:00:00",
        "machineID": 1,
        "volt": 147.0,
        "rotate": 1028.0,
        "pressure": 94.8,
        "vibration": 53.2
      },
      {
        "datetime": "2025-05-23T14:00:00",
        "machineID": 1,
        "volt": 146.0,
        "rotate": 1024.0,
        "pressure": 94.4,
        "vibration": 53.6
      },
      {
        "datetime": "2025-05-23T15:00:00",
        "machineID": 1,
        "volt": 145.0,
        "rotate": 1020.0,
        "pressure": 94.0,
        "vibration": 54.0
      },
      {
        "datetime": "2025-05-23T16:00:00",
        "machineID": 1,
        "volt": 144.0,
        "rotate": 1016.0,
        "pressure": 93.6,
        "vibration": 54.4
      },
      {
        "datetime": "2025-05-23T17:00:00",
        "machineID": 1,
        "volt": 143.0,
        "rotate": 1012.0,
        "pressure": 93.2,
        "vibration": 54.8
      },
      {
        "datetime": "2025-05-23T18:00:00",
        "machineID": 1,
        "volt": 142.0,
        "rotate": 1008.0,
        "pressure": 92.8,
        "vibration": 55.2
      },
      {
        "datetime": "2025-05-23T19:00:00",
        "machineID": 1,
        "volt": 141.0,
        "rotate": 1004.0,
        "pressure": 92.4,
        "vibration": 55.6
      },
      {
        "datetime": "2025-05-23T20:00:00",
        "machineID": 1,
        "volt": 140.0,
        "rotate": 1000.0,
        "pressure": 92.0,
        "vibration": 56.0
      },
      {
        "datetime": "2025-05-23T21:00:00",
        "machineID": 1,
        "volt": 139.0,
        "rotate": 996.0,
        "pressure": 91.6,
        "vibration": 56.4
      },
      {
        "datetime": "2025-05-23T22:00:00",
        "machineID": 1,
        "volt": 138.0,
        "rotate": 992.0,
        "pressure": 91.2,
        "vibration": 56.8
      },
      {
        "datetime": "2025-05-23T23:00:00",
        "machineID": 1,
        "volt": 137.0,
        "rotate": 988.0,
        "pressure": 90.8,
        "vibration": 57.2
      },
      {
        "datetime": "2025-05-23T00:00:00",
        "machineID": 1,
        "volt": 170.0,
        "rotate": 1085.0,
        "pressure": 103.0,
        "vibration": 47.0
      }
    ],
    "errorsLast24h": [
      {
        "datetime": "2025-05-22T05:00:00",
        "machineID": 1,
        "errorID": "error1"
      },
      {
        "datetime": "2025-05-23T23:00:00",
        "machineID": 1,
        "errorID": "error4"
      }
    ]
  }
        ]
    },
    "multiple_machines": {
        "summary": "Multiple machines example",
        "description": "Predict failure for two machines.",
        "value": [
            {
                "machineID": 1,
                "telemetryLast24h": [
                    {"datetime": "2025-05-24T08:00:00Z", "machineID": 1, "volt": 175.0, "rotate": 415.5, "pressure": 112.0, "vibration": 44.0}
                ],
                "errorsLast24h": []
            },
            {
                "machineID": 2,
                "telemetryLast24h": [
                    {"datetime": "2025-05-24T09:00:00Z", "machineID": 2, "volt": 180.0, "rotate": 400.0, "pressure": 120.0, "vibration": 40.0}
                ],
                "errorsLast24h": [
                     {"datetime": "2025-05-24T09:15:00Z", "machineID": 2, "errorID": "error3"}
                ]
            }
        ]
    }
}
@router.post("/predict", 
             response_model=PredictionResponse,
             summary="Predict Failure Risk",
             description="Receives telemetry and error data for one or more machines over the last 24 hours and returns the predicted risk of failure for each machine.",
             tags=["Predictions"],
             openapi_extra={
                 "requestBody": {
                     "content": {
                         "application/json": {
                             "examples": api_examples 
                         }
                     }
                 }
             }
            )
async def predict_failure_risk(
    payload: PredictionInput = Body(...) 
):
    """
    Endpoint to predict failure risk based on the last 24 hours of telemetry and error data.
    """
    logger.info(f"Received prediction request for {len(payload.root)} machines.")
    if not payload.root:
        raise HTTPException(status_code=400, detail="Request body cannot be empty.")

    try:
        predictions_raw = batch_predict(payload.root)
        response_data = [PredictionOutputRecord(**p) for p in predictions_raw]
        return PredictionResponse(root=response_data)

    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")
