from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

# Cargar el modelo entrenado
model = load('model/flights-jan-v1.joblib')

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])

class InputData(BaseModel):
    DAY_OF_WEEK: int
    DEP_DEL15: int

class PredictionResult(BaseModel):
    predicted_fly: bool

@app.post("/predict", response_model=PredictionResult)
def predict(data: InputData):
    # Crear un DataFrame con los datos de entrada
    input_data = pd.DataFrame([data.dict()])
    
    # Realizar la predicción utilizando el modelo cargado
    prediction = model.predict(input_data)
    
    # Obtener la etiqueta de la predicción
    predicted_fly = bool(prediction[0])

    # Devolver la predicción como respuesta JSON
    return {"predicted_fly": predicted_fly}
