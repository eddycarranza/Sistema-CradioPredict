"""
Script de entrenamiento standalone para el modelo de predicción de enfermedad
cardíaca. Condensa en un solo pipeline lineal los pasos de limpieza, balanceo
y entrenamiento que están explorados con más detalle (EDA, PCA, SHAP, LIME,
etc.) en el notebook heart_disease_cov03_EJECUTADO.ipynb.

Uso:
    pip install -r requirements.txt
    python train.py

Genera (en model/):
    modelo_heart_disease.joblib
    pipeline_preprocesamiento.joblib
    feature_columns.json

Estos son los mismos artefactos que ya usa api/index.py, así que correr este
script vuelve a generarlos desde cero (útil si cambia el dataset o quieres
reentrenar con otros hiperparámetros).
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "heart_disease_uci.csv"
MODEL_DIR = BASE_DIR / "model"
RANDOM_STATE = 42

pd.set_option("future.no_silent_downcasting", True)


# ---------------------------------------------------------------------------
# 1. Carga y limpieza (replica las secciones 2.1-2.5 del notebook)
# ---------------------------------------------------------------------------
def cargar_y_limpiar(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop(columns=["id"], errors="ignore")

    # Imputación: mediana para numéricas sesgadas, moda para categóricas
    for col in ["trestbps", "chol", "thalch", "oldpeak"]:
        df[col] = df[col].fillna(df[col].median())
    for col in ["fbs", "exang", "restecg", "slope", "thal"]:
        df[col] = df[col].fillna(df[col].mode()[0])
    df["ca"] = df["ca"].fillna(df["ca"].median())

    df["ca"] = df["ca"].astype(int)
    df["fbs"] = df["fbs"].astype(bool)
    df["exang"] = df["exang"].astype(bool)

    # Valores fisiológicamente imposibles (chol=0, trestbps=0) -> mediana
    df.loc[df["chol"] == 0, "chol"] = df["chol"].median()
    df.loc[df["trestbps"] == 0, "trestbps"] = df["trestbps"].median()

    # Recorte de outliers (winsorizing) según límites fisiológicos
    df.loc[df["trestbps"] > 200, "trestbps"] = 200
    df.loc[df["chol"] > 500, "chol"] = 500
    df.loc[df["thalch"] < 60, "thalch"] = 60
    df.loc[df["oldpeak"] > 5, "oldpeak"] = 5
    df.loc[df["oldpeak"] < 0, "oldpeak"] = 0

    # Variable objetivo binaria + eliminación de duplicados
    df["heart_disease"] = (df["num"] > 0).astype(int)
    df = df.drop_duplicates()

    return df


# ---------------------------------------------------------------------------
# 2. Balanceo de clases (submuestreo de la clase mayoritaria)
#    NOTA: a diferencia de una versión anterior del notebook, aquí se
#    identifica la clase mayoritaria/minoritaria por conteo real, no por
#    posición de value_counts() (ese era un bug ya corregido).
# ---------------------------------------------------------------------------
def balancear_clases(df: pd.DataFrame) -> pd.DataFrame:
    count_0 = int((df["heart_disease"] == 0).sum())
    count_1 = int((df["heart_disease"] == 1).sum())
    clase_0 = df[df["heart_disease"] == 0]
    clase_1 = df[df["heart_disease"] == 1]

    if count_1 >= count_0:
        mayoritaria, minoritaria = clase_1, clase_0
    else:
        mayoritaria, minoritaria = clase_0, clase_1

    mayoritaria_downsampled = resample(
        mayoritaria, replace=False, n_samples=len(minoritaria), random_state=27
    )
    return pd.concat([minoritaria, mayoritaria_downsampled])


# ---------------------------------------------------------------------------
# 3. Codificación one-hot (mismas columnas/orden que espera api/index.py)
# ---------------------------------------------------------------------------
def codificar_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df.drop(columns=["heart_disease", "num", "dataset_source"], errors="ignore")
    X = pd.get_dummies(X, columns=["sex", "dataset", "cp", "restecg", "slope", "thal"], drop_first=True)
    return X


def main():
    print("1. Cargando y limpiando datos...")
    df = cargar_y_limpiar(DATA_PATH)
    print(f"   {df.shape[0]} registros tras limpieza y de-duplicación")

    print("2. Balanceando clases...")
    df_bal = balancear_clases(df)
    print(f"   {df_bal.shape[0]} registros balanceados "
          f"({df_bal['heart_disease'].value_counts().to_dict()})")

    print("3. Codificando variables categóricas...")
    X_raw = codificar_features(df_bal)
    y = df_bal["heart_disease"]

    X_train_r, X_test_r, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    preproc = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    X_train = preproc.fit_transform(X_train_r)
    X_test = preproc.transform(X_test_r)

    print("4. Entrenando y comparando modelos (Logistic, Tree, RF, SVM) + benchmark...")
    dummy = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
    dummy.fit(X_train, y_train)

    modelos = {
        "Logistic Regression": LogisticRegression(random_state=RANDOM_STATE, max_iter=1500),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "SVM": SVC(probability=True, random_state=RANDOM_STATE),
    }

    resultados = {}
    for nombre, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        y_proba = modelo.predict_proba(X_test)[:, 1]
        resultados[nombre] = {
            "modelo": modelo,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "auc": roc_auc_score(y_test, y_proba),
        }
        print(f"   {nombre:22s} AUC={resultados[nombre]['auc']:.4f}  F1={resultados[nombre]['f1']:.4f}")

    print("5. GridSearchCV sobre Random Forest (validación cruzada 5-fold)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    grid_rf = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE),
        param_grid={
            "n_estimators": [100, 120, 200],
            "max_depth": [10, 15, 20],
            "min_samples_split": [10, 15, 20],
            "criterion": ["gini", "entropy"],
        },
        scoring="f1",
        cv=cv,
        n_jobs=2,
    )
    grid_rf.fit(X_train, y_train)
    print(f"   Mejor F1 (CV): {grid_rf.best_score_:.4f}  params={grid_rf.best_params_}")

    rf_tuned = grid_rf.best_estimator_
    y_pred_tuned = rf_tuned.predict(X_test)
    y_proba_tuned = rf_tuned.predict_proba(X_test)[:, 1]
    resultados["Random Forest (GridSearchCV)"] = {
        "modelo": rf_tuned,
        "accuracy": accuracy_score(y_test, y_pred_tuned),
        "precision": precision_score(y_test, y_pred_tuned),
        "recall": recall_score(y_test, y_pred_tuned),
        "f1": f1_score(y_test, y_pred_tuned),
        "auc": roc_auc_score(y_test, y_proba_tuned),
    }

    print("6. GridSearchCV sobre Logistic Regression (validación cruzada 5-fold)...")
    grid_lr = GridSearchCV(
        LogisticRegression(random_state=RANDOM_STATE),
        param_grid={
            "C": [0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 1, 2, 5, 10],
            "penalty": ["l2"],
            "solver": ["lbfgs"],
            "max_iter": [2000],
        },
        scoring="roc_auc",
        cv=cv,
        n_jobs=2,
    )
    grid_lr.fit(X_train, y_train)
    print(f"   Mejor AUC (CV): {grid_lr.best_score_:.4f}  params={grid_lr.best_params_}")

    lr_tuned = grid_lr.best_estimator_
    y_pred_lr_tuned = lr_tuned.predict(X_test)
    y_proba_lr_tuned = lr_tuned.predict_proba(X_test)[:, 1]
    resultados["Logistic Regression (GridSearchCV)"] = {
        "modelo": lr_tuned,
        "accuracy": accuracy_score(y_test, y_pred_lr_tuned),
        "precision": precision_score(y_test, y_pred_lr_tuned),
        "recall": recall_score(y_test, y_pred_lr_tuned),
        "f1": f1_score(y_test, y_pred_lr_tuned),
        "auc": roc_auc_score(y_test, y_proba_lr_tuned),
    }
    print(f"   Test -> Accuracy={resultados['Logistic Regression (GridSearchCV)']['accuracy']:.4f}  "
          f"AUC={resultados['Logistic Regression (GridSearchCV)']['auc']:.4f}")
    print("   Nota: la mejora sobre el modelo por defecto suele ser marginal en este dataset "
          "(820 filas balanceadas) — es de todos modos la elección de C metodológicamente "
          "correcta porque se valida con 5-fold en vez de una sola partición train/test.")

    mejor_nombre = max(resultados, key=lambda k: resultados[k]["auc"])
    modelo_final = resultados[mejor_nombre]["modelo"]
    print(f"\n7. Mejor modelo según AUC-ROC: {mejor_nombre} "
          f"(AUC={resultados[mejor_nombre]['auc']:.4f}, F1={resultados[mejor_nombre]['f1']:.4f})")

    print("8. Guardando artefactos en model/...")
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(modelo_final, MODEL_DIR / "modelo_heart_disease.joblib")
    joblib.dump(preproc, MODEL_DIR / "pipeline_preprocesamiento.joblib")
    with open(MODEL_DIR / "feature_columns.json", "w") as f:
        json.dump(list(X_raw.columns), f)

    print(f"   Modelo:   {MODEL_DIR / 'modelo_heart_disease.joblib'}")
    print(f"   Pipeline: {MODEL_DIR / 'pipeline_preprocesamiento.joblib'}")
    print(f"   Columnas: {MODEL_DIR / 'feature_columns.json'} ({len(X_raw.columns)} columnas)")
    print("\nListo. api/index.py ya puede usar estos artefactos directamente.")


if __name__ == "__main__":
    main()
