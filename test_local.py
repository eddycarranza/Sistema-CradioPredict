"""
Prueba rápida de la API en local, sin necesidad de desplegar en Vercel.

Uso:
    pip install -r requirements.txt httpx
    python test_local.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "api"))

from fastapi.testclient import TestClient  # noqa: E402
from index import app  # noqa: E402

client = TestClient(app)

paciente_ejemplo = {
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

if __name__ == "__main__":
    r = client.get("/api/status")
    print("GET /api/status ->", r.status_code, r.json())

    r = client.post("/predict", json=paciente_ejemplo)
    print("POST /predict ->", r.status_code, r.json())

    assert r.status_code == 200
    print("\nOK: la API responde correctamente.")
