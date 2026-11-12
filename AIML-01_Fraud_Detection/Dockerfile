FROM python:3.10-slim

WORKDIR /app

# Install runtime dependencies with high timeout and retries for network resilience
RUN pip install --no-cache-dir --default-timeout=1000 --retries=10 \
    fastapi \
    uvicorn \
    pandas \
    scikit-learn \
    xgboost \
    kafka-python

# Copy app files (venv, csv, mlruns ignored by .dockerignore)
COPY . .

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
