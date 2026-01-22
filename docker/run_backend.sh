#!/bin/bash
set -e

# Wait for MLflow if needed
echo "Waiting for MLflow..."
while ! nc -z mlflow 5050; do
  sleep 1
done

echo "MLflow is ready, starting backend..."

# Start the application
exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}