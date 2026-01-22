"""
Data Processing Module for CKD Detection Project
Handles data cleaning, analysis, and visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, List, Tuple
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Visualization Config
plt.style.use('ggplot')
sns.set_context('notebook')


class DataProcessor:
    """Handles data cleaning, analysis and visualization"""
    
    def __init__(self, data_path: str = 'data/processed/ckd_normalized.csv'):
        """
        Initialize DataProcessor
        
        Args:
            data_path: Path to loaded data CSV file
        """
        self.data_path = Path(data_path)
        self.df = None
        self.reports_dir = Path('reports')
        self.figures_dir = Path('figures')
        
        # Create output directories
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
    
    def load_data(self) -> pd.DataFrame:
        """Load processed data"""
        try:
            self.df = pd.read_csv(self.data_path)
            logger.info(f"Data loaded successfully. Shape: {self.df.shape}")
            return self.df
        except FileNotFoundError:
            logger.error(f"Error: File not found at {self.data_path}")
            raise
    
    def clean_data(self) -> pd.DataFrame:
        """
        Clean data by removing whitespace and standardizing values
        
        Returns:
            Cleaned DataFrame
        """
        logger.info("Starting data cleaning...")
        
        # Columns with known issues
        cols_to_clean = ['dm', 'status']
        
        def clean_text(x):
            if isinstance(x, str):
                return x.strip().lower()
            return x
        
        for col in cols_to_clean:
            if col in self.df.columns:
                self.df[col] = self.df[col].apply(clean_text)
        
        # Verify cleaning
        logger.info("Data cleaning completed")
        logger.info(f"Status unique values: {self.df['status'].unique()}")
        
        return self.df
    
    def analyze_missing_values(self) -> pd.DataFrame:
        """
        Analyze and report missing values
        
        Returns:
            DataFrame with missing value statistics
        """
        logger.info("Analyzing missing values...")
        
        missing_count = self.df.isnull().sum().sort_values(ascending=False)
        missing_percent = (self.df.isnull().sum() / len(self.df)) * 100
        missing_data = pd.concat(
            [missing_count, missing_percent], 
            axis=1, 
            keys=['Total', 'Percent']
        )
        
        # Filter only columns with missing values
        missing_data = missing_data[missing_data['Total'] > 0]
        
        # Save report
        report_path = self.reports_dir / 'missing_values_report.csv'
        missing_data.to_csv(report_path)
        logger.info(f"Missing values report saved to {report_path}")
        
        # Create visualization
        self._plot_missing_values()
        
        return missing_data
    
    def _plot_missing_values(self):
        """Create missing values heatmap"""
        plt.figure(figsize=(12, 6))
        sns.heatmap(self.df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
        plt.title('Missing Values Heatmap')
        plt.tight_layout()
        
        fig_path = self.figures_dir / 'missing_values_heatmap.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Missing values heatmap saved to {fig_path}")
    
    def generate_statistical_summary(self) -> Dict:
        """
        Generate statistical summaries for numerical and categorical features
        
        Returns:
            Dictionary with summary statistics
        """
        logger.info("Generating statistical summaries...")
        
        # Numerical summary
        numeric_summary = self.df.describe().T
        numeric_path = self.reports_dir / 'numeric_summary.csv'
        numeric_summary.to_csv(numeric_path)
        
        # Categorical summary
        categorical_summary = self.df.describe(include=['object']).T
        categorical_path = self.reports_dir / 'categorical_summary.csv'
        categorical_summary.to_csv(categorical_path)
        
        logger.info(f"Summaries saved to {self.reports_dir}")
        
        return {
            'numeric': numeric_summary,
            'categorical': categorical_summary
        }
    
    def plot_univariate_numeric(self):
        """Create distribution plots for numerical features"""
        logger.info("Creating univariate plots for numerical features...")
        
        numeric_cols_mapping = {
            'age': 'Age',
            'bp': 'Blood Pressure',
            'bgr': 'Blood Glucose Random',
            'bu': 'Blood Urea',
            'sc': 'Serum Creatinine',
            'sod': 'Sodium',
            'pot': 'Potassium',
            'hemo': 'Hemoglobin',
            'pcv': 'Packed Cell Volume',
            'wbcc': 'White Blood Cell Count',
            'rbcc': 'Red Blood Cell Count'
        }
        
        fig, axes = plt.subplots(4, 3, figsize=(18, 16))
        axes = axes.flatten()
        
        for i, (col, full_name) in enumerate(numeric_cols_mapping.items()):
            if col in self.df.columns:
                sns.histplot(
                    self.df[col].dropna(), 
                    kde=True, 
                    ax=axes[i], 
                    color='skyblue',
                    edgecolor='black'
                )
                axes[i].set_title(f'Distribution of {full_name}', fontsize=12, fontweight='bold')
                axes[i].set_xlabel(full_name)
                axes[i].set_ylabel('Count')
        
        # Remove unused subplots
        for j in range(len(numeric_cols_mapping), len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        fig_path = self.figures_dir / 'numeric_distributions.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Numerical distributions saved to {fig_path}")
    
    def plot_target_distribution(self):
        """Plot target variable distribution"""
        logger.info("Creating target distribution plot...")
        
        plt.figure(figsize=(6, 4))
        sns.countplot(x='status', data=self.df, palette='viridis', hue='status', legend=False)
        plt.title('Class Distribution (CKD vs Not CKD)')
        plt.tight_layout()
        
        fig_path = self.figures_dir / 'target_distribution.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Target distribution saved to {fig_path}")
    
    def plot_univariate_categorical(self):
        """Create distribution plots for categorical features"""
        logger.info("Creating univariate plots for categorical features...")
        
        categorical_cols_mapping = {
            'sg': 'Specific Gravity',
            'al': 'Albumin',
            'su': 'Sugar',
            'rbc': 'Red Blood Cells',
            'pc': 'Pus Cell',
            'pcc': 'Pus Cell Clumps',
            'ba': 'Bacteria',
            'htn': 'Hypertension',
            'dm': 'Diabetes Mellitus',
            'cad': 'Coronary Artery Disease',
            'appet': 'Appetite',
            'pe': 'Pedal Edema',
            'ane': 'Anemia'
        }
        
        fig, axes = plt.subplots(5, 3, figsize=(18, 20))
        axes = axes.flatten()
        
        for i, (col, full_name) in enumerate(categorical_cols_mapping.items()):
            if col in self.df.columns:
                order = sorted(self.df[col].dropna().unique()) if col in ['sg', 'al', 'su'] else None
                sns.countplot(
                    x=col, 
                    data=self.df, 
                    ax=axes[i], 
                    palette='viridis', 
                    order=order,
                    hue=col,
                    legend=False
                )
                axes[i].set_title(f'Distribution of {full_name}', fontsize=12, fontweight='bold')
                axes[i].set_xlabel(full_name)
                axes[i].set_ylabel('Count')
        
        # Remove empty subplots
        for j in range(len(categorical_cols_mapping), len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        fig_path = self.figures_dir / 'categorical_distributions.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Categorical distributions saved to {fig_path}")
    
    def plot_correlation_matrix(self):
        """Create correlation matrix heatmap"""
        logger.info("Creating correlation matrix...")
        
        plt.figure(figsize=(12, 10))
        corr_matrix = self.df.select_dtypes(include=[np.number]).corr()
        
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Matrix')
        plt.tight_layout()
        
        fig_path = self.figures_dir / 'correlation_matrix.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Correlation matrix saved to {fig_path}")
        
        # Save correlation matrix as CSV
        corr_path = self.reports_dir / 'correlation_matrix.csv'
        corr_matrix.to_csv(corr_path)
        logger.info(f"Correlation matrix data saved to {corr_path}")
    
    def save_cleaned_data(self, output_path: str = 'data/processed/cleaned_data.csv'):
        """
        Save cleaned data
        
        Args:
            output_path: Path to save cleaned data
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.df.to_csv(output_path, index=False)
        logger.info(f"Cleaned data saved to {output_path}")
    
    def generate_data_quality_report(self) -> Dict:
        """
        Generate comprehensive data quality report
        
        Returns:
            Dictionary with quality metrics
        """
        logger.info("Generating data quality report...")
        
        report = {
            'total_records': len(self.df),
            'total_features': len(self.df.columns),
            'numeric_features': len(self.df.select_dtypes(include=[np.number]).columns),
            'categorical_features': len(self.df.select_dtypes(include=['object']).columns),
            'total_missing_values': self.df.isnull().sum().sum(),
            'missing_percentage': (self.df.isnull().sum().sum() / (self.df.shape[0] * self.df.shape[1])) * 100,
            'duplicate_records': self.df.duplicated().sum(),
            'target_distribution': self.df['status'].value_counts().to_dict()
        }
        
        # Save report as JSON
        report_path = self.reports_dir / 'data_quality_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
        
        logger.info(f"Data quality report saved to {report_path}")
        
        return report


def main():
    """Main execution function"""
    logger.info("Starting data processing pipeline...")
    
    # Initialize processor
    processor = DataProcessor()
    
    # Load data
    processor.load_data()
    
    # Clean data
    processor.clean_data()
    
    # Analyze missing values
    missing_stats = processor.analyze_missing_values()
    print("\n=== Missing Values Summary ===")
    print(missing_stats)
    
    # Generate statistical summaries
    summaries = processor.generate_statistical_summary()
    
    # Create visualizations
    processor.plot_univariate_numeric()
    processor.plot_target_distribution()
    processor.plot_univariate_categorical()
    processor.plot_correlation_matrix()
    
    # Generate quality report
    quality_report = processor.generate_data_quality_report()
    print("\n=== Data Quality Report ===")
    print(json.dumps(quality_report, indent=2))
    
    # Save cleaned data
    processor.save_cleaned_data()
    
    logger.info("Data processing pipeline completed successfully!")
    logger.info(f"Reports saved to: {processor.reports_dir}")
    logger.info(f"Figures saved to: {processor.figures_dir}")


if __name__ == "__main__":
    main()