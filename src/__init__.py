"""
CKD Detection Project
MLOps pipeline for Chronic Kidney Disease detection
"""

__version__ = '1.0.0'
__author__ = 'CKD Detection Team'

from .data_loading import DataLoader
from .data_processing import DataProcessor

__all__ = ['DataLoader', 'DataProcessor']