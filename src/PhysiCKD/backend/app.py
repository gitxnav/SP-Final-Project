from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os

app = Flask(__name__)
CORS(app)

# Load the model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'ckd_final_model_full_data_gradient_boosting.pkl')
try:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Load the dataset for comparison statistics
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'ckd_imputed.csv')
dataset = None
comparison_stats = None

try:
    dataset = pd.read_csv(DATA_PATH)
    print(f"Dataset loaded successfully from {DATA_PATH}")
    
    # Calculate comparison statistics for each feature grouped by CKD status
    feature_names = ['hemo', 'sg', 'sc', 'rbcc', 'pcv', 'htn', 'dm', 'bp', 'age']
    
    # Categorical features that should use mode instead of mean
    categorical_features = ['dm', 'htn', 'sg']
    
    # Map status to binary: 'ckd' -> 1, 'notckd' -> 0
    dataset['ckd_status'] = (dataset['status'] == 'ckd').astype(int)
    
    comparison_stats = {}
    for feature in feature_names:
        if feature in dataset.columns:
            ckd_data = dataset[dataset['ckd_status'] == 1][feature]
            notckd_data = dataset[dataset['ckd_status'] == 0][feature]
            
            is_categorical = feature in categorical_features
            
            if is_categorical:
                # For categorical features, use mode
                ckd_mode = ckd_data.mode()
                notckd_mode = notckd_data.mode()
                
                comparison_stats[feature] = {
                    'ckd': {
                        'mode': float(ckd_mode.iloc[0]) if len(ckd_mode) > 0 else None,
                        'mean': None,
                        'median': None,
                        'std': None,
                        'min': float(ckd_data.min()),
                        'max': float(ckd_data.max())
                    },
                    'notckd': {
                        'mode': float(notckd_mode.iloc[0]) if len(notckd_mode) > 0 else None,
                        'mean': None,
                        'median': None,
                        'std': None,
                        'min': float(notckd_data.min()),
                        'max': float(notckd_data.max())
                    }
                }
            else:
                # For numerical features, use mean, median, std
                comparison_stats[feature] = {
                    'ckd': {
                        'mean': float(ckd_data.mean()),
                        'median': float(ckd_data.median()),
                        'std': float(ckd_data.std()),
                        'min': float(ckd_data.min()),
                        'max': float(ckd_data.max()),
                        'mode': None
                    },
                    'notckd': {
                        'mean': float(notckd_data.mean()),
                        'median': float(notckd_data.median()),
                        'std': float(notckd_data.std()),
                        'min': float(notckd_data.min()),
                        'max': float(notckd_data.max()),
                        'mode': None
                    }
                }
    
    print("Comparison statistics calculated successfully")
except Exception as e:
    print(f"Error loading dataset: {e}")
    comparison_stats = None

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        data = request.json
        
        # Ensure input data has the correct feature names and order
        feature_names = ['hemo', 'sg', 'sc', 'rbcc', 'pcv', 'htn', 'dm', 'bp', 'age']
        
        # Check if all required features are present
        if not all(feature in data for feature in feature_names):
            return jsonify({'error': f'Missing features. Expected: {feature_names}'}), 400

        # Create DataFrame with a single row, strictly ordering columns
        input_data = {feature: [data[feature]] for feature in feature_names}
        input_df = pd.DataFrame(input_data, columns=feature_names)

        # Predict
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][prediction]

        # Prepare patient input values for comparison
        patient_values = {feature: float(data[feature]) for feature in feature_names}
        
        # Prepare comparison data
        comparison_data = {}
        categorical_features = ['dm', 'htn', 'sg']
        
        if comparison_stats:
            for feature in feature_names:
                if feature in comparison_stats:
                    is_categorical = feature in categorical_features
                    
                    if is_categorical:
                        # For categorical features, use mode
                        comparison_data[feature] = {
                            'patient_value': patient_values[feature],
                            'ckd_mode': comparison_stats[feature]['ckd']['mode'],
                            'notckd_mode': comparison_stats[feature]['notckd']['mode'],
                            'ckd_mean': None,
                            'ckd_std': None,
                            'notckd_mean': None,
                            'notckd_std': None,
                            'ckd_median': None,
                            'notckd_median': None
                        }
                    else:
                        # For numerical features, use mean, std, median
                        comparison_data[feature] = {
                            'patient_value': patient_values[feature],
                            'ckd_mean': comparison_stats[feature]['ckd']['mean'],
                            'ckd_std': comparison_stats[feature]['ckd']['std'],
                            'notckd_mean': comparison_stats[feature]['notckd']['mean'],
                            'notckd_std': comparison_stats[feature]['notckd']['std'],
                            'ckd_median': comparison_stats[feature]['ckd']['median'],
                            'notckd_median': comparison_stats[feature]['notckd']['median'],
                            'ckd_mode': None,
                            'notckd_mode': None
                        }

        result = {
            'class': int(prediction),
            'confidence': float(probability),
            'prediction_text': 'CKD' if prediction == 1 else 'NO CKD',
            'patient_values': patient_values,
            'comparison_data': comparison_data
        }
        
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
