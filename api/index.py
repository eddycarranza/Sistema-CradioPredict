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
THRESHOLDS_PATH = BASE_DIR / "model" / "decision_thresholds.json"

modelo = joblib.load(MODEL_PATH)
pipeline = joblib.load(PIPELINE_PATH)
FEATURE_COLUMNS: list[str] = json.load(open(COLUMNS_PATH))

# Umbrales de decision (probabilidad -> clase) calibrados por sexo. Es la
# medida de mitigacion de sesgos del proyecto: el analisis por subgrupo
# (seccion 8.4/11.2 del informe) mostro que, con el umbral por defecto de
# 0.5, el Recall en mujeres era notablemente menor que en hombres. En vez
# de reentrenar el modelo, se calibro (en train.py, paso 9) un umbral mas
# bajo para el grupo en desventaja, exigiendo que su precision no empeore
# respecto al umbral por defecto. Ver model/decision_thresholds.json.
try:
    DECISION_THRESHOLDS: dict = json.load(open(THRESHOLDS_PATH))
except FileNotFoundError:
    DECISION_THRESHOLDS = {"_default": 0.5}

# Nombres legibles de cada columna, para mostrar la explicación al usuario final
FEATURE_LABELS = {
    "age": "Edad",
    "trestbps": "Presión arterial en reposo",
    "chol": "Colesterol sérico",
    "fbs": "Glucemia en ayunas alta",
    "thalch": "Frecuencia cardíaca máxima",
    "exang": "Angina inducida por ejercicio",
    "oldpeak": "Depresión del segmento ST",
    "ca": "N° de vasos coloreados",
    "sex_Male": "Sexo masculino",
    "dataset_Hungary": "Institución: Hungría",
    "dataset_Switzerland": "Institución: Suiza",
    "dataset_VA Long Beach": "Institución: VA Long Beach",
    "cp_atypical angina": "Dolor de pecho: angina atípica",
    "cp_non-anginal": "Dolor de pecho: no anginoso",
    "cp_typical angina": "Dolor de pecho: angina típica",
    "restecg_normal": "ECG en reposo: normal",
    "restecg_st-t abnormality": "ECG en reposo: anomalía ST-T",
    "slope_flat": "Pendiente del ST: plana",
    "slope_upsloping": "Pendiente del ST: ascendente",
    "thal_normal": "Talasemia: normal",
    "thal_reversable defect": "Talasemia: defecto reversible",
}

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


class Factor(BaseModel):
    variable: str
    contribucion: float
    direccion: Literal["aumenta", "reduce"]


class Prediccion(BaseModel):
    prediccion: int
    diagnostico: str
    probabilidad_enfermedad: float
    factores: list[Factor] = []
    umbral_aplicado: float
    nota_umbral: str


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
@app.get("/api/status")
def status():
    return {
        "status": "ok",
        "modelo": type(modelo).__name__,
        "n_features": len(FEATURE_COLUMNS),
        "umbrales_decision": DECISION_THRESHOLDS,
    }


@app.get("/api/health")
def health():
    return {"status": "healthy"}


@app.post("/api/predict", response_model=Prediccion)
def predict(paciente: Paciente):
    try:
        X = preparar_input(paciente)
        X_proc = pipeline.transform(X)
        proba = float(modelo.predict_proba(X_proc)[0][1])

        # Umbral calibrado por sexo (mitigacion de sesgo, ver comentario arriba).
        # Si el grupo del paciente no tiene un umbral especifico calibrado,
        # se usa el umbral por defecto (0.5).
        umbral = DECISION_THRESHOLDS.get(paciente.sex, DECISION_THRESHOLDS.get("_default", 0.5))
        pred = int(proba >= umbral)

        # ---------------------------------------------------------------
        # Explicación real de la predicción: para un modelo lineal
        # (Regresión Logística), la contribución exacta de cada variable
        # al resultado es coeficiente_i * valor_escalado_i. No es una
        # aproximación tipo SHAP/LIME: es el cálculo real que hace el
        # modelo internamente, expuesto variable por variable.
        # ---------------------------------------------------------------
        factores: list[Factor] = []
        if hasattr(modelo, "coef_"):
            contribuciones = modelo.coef_[0] * X_proc[0]
            top_idx = sorted(
                range(len(contribuciones)),
                key=lambda i: abs(contribuciones[i]),
                reverse=True,
            )[:6]
            for i in top_idx:
                col = FEATURE_COLUMNS[i]
                valor = float(contribuciones[i])
                factores.append(Factor(
                    variable=FEATURE_LABELS.get(col, col),
                    contribucion=round(valor, 4),
                    direccion="aumenta" if valor > 0 else "reduce",
                ))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Error al procesar la predicción: {exc}") from exc

    return Prediccion(
        prediccion=pred,
        diagnostico="Con enfermedad cardíaca" if pred == 1 else "Sin enfermedad cardíaca",
        probabilidad_enfermedad=round(proba, 4),
        factores=factores,
        umbral_aplicado=umbral,
        nota_umbral=(
            "Umbral calibrado por sexo como medida de mitigación de sesgo: "
            "el análisis por subgrupo mostró menor Recall en mujeres con el "
            "umbral por defecto (0.5), por lo que se ajustó para ese grupo "
            "sin reducir su precisión."
        ) if umbral != 0.5 else "Umbral por defecto (0.5); no se detectó necesidad de calibración para este grupo.",
    )
