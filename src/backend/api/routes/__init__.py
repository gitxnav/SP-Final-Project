"""
API Routes Package
"""

from . import health, prediction, mlflow, training

__all__ = ["health", "training", "mlflow", "prediction"]