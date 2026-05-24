# NeuroPlay Predictive Interface

Interfaz profesional para consultar el modelo entrenado de NeuroPlay.

## Archivos

- `app.py`: API FastAPI que carga el modelo `.pkl`.
- `static/index.html`: interfaz web.
- `static/style.css`: estilos visuales.
- `neuroplay_modelo_entrenado.pkl`: modelo entrenado.
- `requirements.txt`: dependencias.

## Ejecución local

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

Después abre:

```text
http://127.0.0.1:8000
```

## Nota metodológica

La interfaz devuelve una clasificación experimental basada en etiquetas autodeclaradas durante el juego. No representa una medición clínica, psicológica ni emocional objetiva.
