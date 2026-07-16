"""
API de predicción de enfermedad cardíaca.
Carga el modelo entrenado (Regresión Logística) + pipeline de preprocesamiento
generados en heart_disease_cov03_EJECUTADO.ipynb, y expone un endpoint /predict.

Vercel detecta automáticamente la instancia `app` de FastAPI en este archivo
(api/index.py es uno de los entrypoints soportados). No hace falta configurar
rutas manualmente.
"""

import json
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Carga de artefactos (modelo, pipeline de preprocesamiento, columnas)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "modelo_heart_disease.joblib"
PIPELINE_PATH = BASE_DIR / "model" / "pipeline_preprocesamiento.joblib"
COLUMNS_PATH = BASE_DIR / "model" / "feature_columns.json"

modelo = joblib.load(MODEL_PATH)
pipeline = joblib.load(PIPELINE_PATH)
FEATURE_COLUMNS: list[str] = json.load(open(COLUMNS_PATH))

app = FastAPI(
    title="Heart Disease Prediction API",
    description="Predice si un paciente tiene enfermedad cardíaca a partir de indicadores clínicos.",
    version="1.0.0",
)

# Habilita llamadas desde cualquier frontend (ajusta allow_origins si quieres restringirlo)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Esquema de entrada / salida
# ---------------------------------------------------------------------------
class Paciente(BaseModel):
    age: float = Field(..., ge=0, le=120, description="Edad del paciente en años")
    sex: Literal["Male", "Female"]
    dataset: Literal["Cleveland", "Hungary", "Switzerland", "VA Long Beach"] = "Cleveland"
    cp: Literal["typical angina", "atypical angina", "non-anginal", "asymptomatic"]
    trestbps: float = Field(..., description="Presión arterial en reposo (mmHg)")
    chol: float = Field(..., description="Colesterol sérico (mg/dl)")
    fbs: bool = Field(..., description="Glucemia en ayunas > 120 mg/dl")
    restecg: Literal["normal", "st-t abnormality", "lv hypertrophy"]
    thalch: float = Field(..., description="Frecuencia cardíaca máxima alcanzada")
    exang: bool = Field(..., description="Angina inducida por ejercicio")
    oldpeak: float = Field(..., description="Depresión del segmento ST inducida por ejercicio")
    slope: Literal["upsloping", "flat", "downsloping"]
    ca: float = Field(..., ge=0, le=3, description="N° de vasos principales coloreados por fluoroscopia")
    thal: Literal["normal", "fixed defect", "reversable defect"]

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 63,
                "sex": "Male",
                "dataset": "Cleveland",
                "cp": "typical angina",
                "trestbps": 145,
                "chol": 233,
                "fbs": True,
                "restecg": "lv hypertrophy",
                "thalch": 150,
                "exang": False,
                "oldpeak": 2.3,
                "slope": "downsloping",
                "ca": 0,
                "thal": "fixed defect",
            }
        }
    }


class Prediccion(BaseModel):
    prediccion: int
    diagnostico: str
    probabilidad_enfermedad: float


# ---------------------------------------------------------------------------
# Preparación de features: replica EXACTAMENTE el one-hot encoding
# (pd.get_dummies con drop_first=True) usado en el entrenamiento.
# No se usa pd.get_dummies aquí a propósito: aplicado sobre una sola fila
# genera columnas distintas según el valor recibido, lo cual no coincide
# con las columnas fijas que el modelo espera. Por eso se arma la fila
# manualmente contra el listado fijo de FEATURE_COLUMNS.
# ---------------------------------------------------------------------------
def preparar_input(p: Paciente) -> pd.DataFrame:
    fila = {col: 0 for col in FEATURE_COLUMNS}

    fila["age"] = p.age
    fila["trestbps"] = p.trestbps
    fila["chol"] = p.chol
    fila["fbs"] = int(p.fbs)
    fila["thalch"] = p.thalch
    fila["exang"] = int(p.exang)
    fila["oldpeak"] = p.oldpeak
    fila["ca"] = p.ca

    categoricas = {
        "sex_Male": p.sex == "Male",
        f"dataset_{p.dataset}": True,
        f"cp_{p.cp}": True,
        f"restecg_{p.restecg}": True,
        f"slope_{p.slope}": True,
        f"thal_{p.thal}": True,
    }
    for col, activa in categoricas.items():
        if activa and col in fila:
            fila[col] = 1

    return pd.DataFrame([fila])[FEATURE_COLUMNS]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def status():
    return {
        "status": "ok",
        "modelo": type(modelo).__name__,
        "n_features": len(FEATURE_COLUMNS),
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=Prediccion)
def predict(paciente: Paciente):
    try:
        X = preparar_input(paciente)
        X_proc = pipeline.transform(X)
        pred = int(modelo.predict(X_proc)[0])
        proba = float(modelo.predict_proba(X_proc)[0][1])
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Error al procesar la predicción: {exc}") from exc

    return Prediccion(
        prediccion=pred,
        diagnostico="Con enfermedad cardíaca" if pred == 1 else "Sin enfermedad cardíaca",
        probabilidad_enfermedad=round(proba, 4),
    )
