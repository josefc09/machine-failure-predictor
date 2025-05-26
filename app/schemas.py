from pydantic import BaseModel, Field, RootModel
from typing import List, Dict, Any
from datetime import datetime

class TelemetryRecord(BaseModel):
    datetime: datetime
    machineID: int
    volt: float
    rotate: float
    pressure: float
    vibration: float

class ErrorRecord(BaseModel):
    datetime: datetime
    machineID: int
    errorID: str

# Defines the input structure for a single machine, including recent telemetry and errors.
class MachineDataInput(BaseModel):
    # Allows 'machineID' to be used as an alias for 'machineId'.
    machineId: int = Field(..., alias='machineID')
    telemetryLast24h: List[TelemetryRecord]
    errorsLast24h: List[ErrorRecord]

    class Config:
        # Enables assignment using field names in addition to aliases.
        allow_population_by_field_name = True

# Defines the overall input structure, which is a list of machine data.
class PredictionInput(RootModel[List[MachineDataInput]]):
    root: List[MachineDataInput]

# Defines the structure for a single prediction output record.
class PredictionOutputRecord(BaseModel):
    machineId: int
    predictionDate: str
    riskOfFailure: str

# Defines the overall output structure, which is a list of prediction records.
class PredictionResponse(RootModel[List[PredictionOutputRecord]]):
    root: List[PredictionOutputRecord]