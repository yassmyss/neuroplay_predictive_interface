from pathlib import Path
from typing import Literal, Optional

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "neuroplay_modelo_entrenado.pkl"

package = joblib.load(MODEL_PATH)
model = package["model"]
FEATURES = package["features"]
MODEL_NAME = package.get("best_model_name", "Modelo NeuroPlay")


MODE_TO_TASK = {
    "memoria": "memoria_visual",
    "persecucion": "coordinacion",
    "reflejos": "reaccion",
    "stroop": "control_cognitivo",
}


class PredictionInput(BaseModel):
    modo: Literal["memoria", "persecucion", "reflejos", "stroop"] = "reflejos"
    dificultad: int = Field(ge=1, le=5, default=2)
    aciertos: int = Field(ge=0, default=12)
    errores: int = Field(ge=0, default=3)
    interacciones: Optional[int] = Field(default=None, ge=0)
    precision_pct: float = Field(ge=0, le=100, default=80)
    reaccion_ms: float = Field(ge=0, default=1200)
    variabilidad_reaccion: float = Field(ge=0, default=250)


def build_feature_row(data: PredictionInput) -> pd.DataFrame:
    interacciones = data.interacciones
    if interacciones is None or interacciones == 0:
        interacciones = data.aciertos + data.errores

    row = {feature: 0 for feature in FEATURES}

    numeric_values = {
        "aciertos": data.aciertos,
        "errores": data.errores,
        "interacciones": interacciones,
        "precision_pct": data.precision_pct,
        "reaccion_ms": data.reaccion_ms,
        "variabilidad_reaccion": data.variabilidad_reaccion,
        "dificultad": data.dificultad,
    }

    for key, value in numeric_values.items():
        if key in row:
            row[key] = value

    mode_feature = f"modo_{data.modo}"
    if mode_feature in row:
        row[mode_feature] = 1

    task = MODE_TO_TASK.get(data.modo)
    task_feature = f"tipo_tarea_{task}"
    if task_feature in row:
        row[task_feature] = 1

    return pd.DataFrame([row], columns=FEATURES)


def generate_recommendation(label: str, data: PredictionInput, confidence: float) -> str:
    label_norm = label.lower()

    if "cansancio" in label_norm:
        return (
            "El patrón se aproxima a una etiqueta de cansancio reportado. "
            "Puede ser conveniente reducir la dificultad, acortar la siguiente ronda "
            "o realizar una pausa breve antes de continuar."
        )

    if "sobrecarga" in label_norm:
        return (
            "El patrón se aproxima a una etiqueta de sobrecarga reportada. "
            "Se recomienda bajar la dificultad y elegir un modo menos exigente durante la siguiente ronda."
        )

    if "frustr" in label_norm:
        return (
            "El patrón se aproxima a una etiqueta de dificultad o incomodidad percibida. "
            "Una buena opción sería reducir ligeramente la dificultad y priorizar precisión sobre velocidad."
        )

    if "enfoque" in label_norm:
        return (
            "El patrón se aproxima a una etiqueta de enfoque reportado. "
            "La partida muestra un rendimiento estable; puede mantenerse la dificultad o aumentarla de forma gradual."
        )

    if "relax" in label_norm:
        return (
            "El patrón se aproxima a una etiqueta de baja presión o relax reportado. "
            "Puede mantenerse el modo actual o aumentar suavemente el reto si se busca mayor intensidad."
        )

    return (
        "El modelo ha generado una clasificación experimental a partir de las métricas introducidas. "
        "Se recomienda interpretar el resultado como apoyo orientativo, no como evaluación emocional objetiva."
    )


app = FastAPI(
    title="NeuroPlay Predictive Interface",
    description="Interfaz predictiva para clasificar etiquetas reportadas a partir de métricas de juego.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get("/")
def home():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "features": FEATURES,
    }


@app.post("/predict")
def predict(data: PredictionInput):
    X = build_feature_row(data)
    prediction = model.predict(X)[0]

    confidence = None
    probabilities = []

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        classes = list(model.classes_)
        probabilities = [
            {
                "label": str(label),
                "probability": round(float(prob), 4)
            }
            for label, prob in sorted(zip(classes, proba), key=lambda item: item[1], reverse=True)
        ]
        confidence = probabilities[0]["probability"]

    return {
        "model": MODEL_NAME,
        "predicted_label": str(prediction),
        "confidence": confidence,
        "recommendation": generate_recommendation(str(prediction), data, confidence or 0),
        "probabilities": probabilities,
        "disclaimer": (
            "Resultado experimental basado en estados autodeclarados. "
            "No representa una medición clínica, psicológica ni emocional objetiva."
        ),
    }