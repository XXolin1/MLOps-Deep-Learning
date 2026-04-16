from pydantic import BaseModel

class PredictionRequest(BaseModel):
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