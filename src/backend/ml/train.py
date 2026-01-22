import numpy as np
import mlflow
from model import CKDModel

def generate_sample_data(n_samples=1000):
    """Generate synthetic CKD data"""
    np.random.seed(42)
    
    # Features
    age = np.random.randint(20, 90, n_samples)
    bp = np.random.randint(60, 180, n_samples)
    sg = np.random.uniform(1.005, 1.025, n_samples)
    albumin = np.random.randint(0, 6, n_samples)
    sugar = np.random.randint(0, 6, n_samples)
    
    X = np.column_stack([age, bp, sg, albumin, sugar])
    
    # Target (simple rule-based)
    y = ((age > 50) & (bp > 100) & (albumin > 2)).astype(int)
    
    return X, y

if __name__ == "__main__":
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("http://mlflow:5050")
    mlflow.set_experiment("ckd_detection")
    
    print("Generating training data...")
    X, y = generate_sample_data()
    
    print("Training model...")
    model = CKDModel()
    metrics = model.train(X, y)
    
    print("\nTraining Complete!")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"F1 Score: {metrics['f1_score']:.3f}")