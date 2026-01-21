"""
CKD Detection Project
MLOps pipeline for Chronic Kidney Disease detection
"""

__version__ = '1.0.0'
__author__ = 'CKD Detection Team'

from .step01_data_loading import DataLoader
from .step02_data_processing import DataProcessor

__all__ = ['DataLoader', 'DataProcessor']
