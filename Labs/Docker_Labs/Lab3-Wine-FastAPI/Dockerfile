# Stage 1: train the model at build time, so the image ships with it
FROM python:3.12-slim AS model_training

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/train.py src/
RUN python src/train.py


# Stage 2: serve predictions. Only the trained model is copied over, not the training run.
FROM python:3.12-slim AS serving

# Don't write .pyc files, and send print() output straight to `docker logs`
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=/app/model

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --from=model_training /app/model ./model
COPY src/main.py src/

# Run as a non-root user
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

# slim images have no curl, so the health check uses Python
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')"

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
