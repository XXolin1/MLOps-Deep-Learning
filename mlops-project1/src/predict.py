from pydantic import BaseModel, Field
from typing import Optional

class PredictionRequest(BaseModel):
    Target: Optional[int] = 1  # Optional - not needed for prediction
    Income: float
    HighBP: float
    HighChol: float
    CholCheck: float
    Smoker: float
    Stroke: float
    HeartDiseaseorAttack: float
    PhysActivity: float
    Fruits: float
    Veggies: float
    HvyAlcoholConsump: float
    AnyHealthcare: float
    NoDocbcCost: float
    DiffWalk: float
    Sex: float
    Education: float
    BMI: float
    MentHlth: float
    PhysHlth: float
    Age: float
    GenHlth: float