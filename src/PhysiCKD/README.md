# Chronic Kidney Disease (CKD) Prediction System

A comprehensive web application for predicting Chronic Kidney Disease risk using machine learning. This system provides an intuitive interface for healthcare professionals to input patient data and receive detailed risk assessments with visual analytics.

## 🎯 Features

- **Machine Learning Prediction**: Utilizes a Gradient Boosting model trained on clinical data to predict CKD risk
- **Interactive Dashboard**: Beautiful, responsive UI with real-time form validation
- **Comprehensive Visualizations**:
  - Risk score distribution charts
  - Key parameters comparison
  - Risk factors analysis
  - Patient values vs. population averages (CKD vs. Non-CKD)
  - Detailed feature-by-feature comparisons
- **Statistical Analysis**: Compares patient data against population statistics for better context
- **Risk Assessment**: Provides risk scores (0-100%) with confidence levels and risk categorization (Low/Medium/High)

## 🛠️ Tech Stack

### Frontend
- **React 19** - Modern UI library
- **Vite** - Fast build tool and dev server
- **Chart.js** - Interactive data visualizations
- **React Chart.js 2** - React wrapper for Chart.js

### Backend
- **Flask** - Python web framework
- **scikit-learn** - Machine learning library
- **pandas** - Data manipulation and analysis
- **joblib** - Model serialization
- **Flask-CORS** - Cross-origin resource sharing

## 📋 Prerequisites

Before you begin, ensure you have the following installed:

- **Node.js** (v16 or higher) and **npm**
- **Python** (v3.8 or higher) and **pip**
- **Git** (for cloning the repository)

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd Mockup_HTML
```

### 2. Backend Setup

Navigate to the backend directory and set up a virtual environment:

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Frontend Setup

Open a new terminal, navigate to the frontend directory, and install dependencies:

```bash
cd frontend
npm install
```

## 🏃 Running the Application

### Start the Backend Server

From the `backend` directory (with virtual environment activated):

```bash
python app.py
```

The backend server will start on `http://localhost:5000`

### Start the Frontend Development Server

From the `frontend` directory (in a new terminal):

```bash
npm run dev
```

The frontend will be available at `http://localhost:5173` (or the port shown in your terminal)

### Access the Application

Open your browser and navigate to the frontend URL (typically `http://localhost:5173`)

## 📖 How to Use

### 1. Enter Patient Information

Fill out the form with the following patient data:

**Patient Information:**
- **Age** (years): 0-120
- **Blood Pressure** (mmHg): 40-250
- **Diabetes Mellitus**: Yes/No
- **Hypertension**: Yes/No

**Laboratory Results:**
- **Hemoglobin** (g/dL): 3.0-20.0
- **Packed Cell Volume** (%): 10-60
- **Red Blood Cell Count** (million/μL): 1.0-8.0
- **Serum Creatinine** (mg/dL): 0.2-30.0
- **Specific Gravity**: Select from 1.005, 1.010, 1.015, 1.020, or 1.025

### 2. Generate Prediction

Click the **"Generate Prediction"** button to submit the form. The system will:
- Validate all input fields
- Send data to the backend API
- Process the prediction using the machine learning model
- Display comprehensive results

### 3. Review Results

The results dashboard will show:

- **Prediction Result**: Risk score percentage and risk level
- **Statistics Cards**: Prediction, risk level, confidence, and model used
- **Risk Score Distribution**: Visual representation of CKD risk
- **Key Parameters Comparison**: Normalized view of important parameters
- **Risk Factors Analysis**: Breakdown of contributing risk factors
- **Population Comparison**: Patient values compared to CKD and Non-CKD population averages
- **Detailed Feature Comparisons**: Individual charts for each key feature

### 4. Clear and Reset

Use the **"Clear Form"** button to reset all fields and start a new prediction.

## 📁 Project Structure

```
Mockup_HTML/
├── backend/
│   ├── app.py                 # Flask application and API endpoints
│   ├── requirements.txt       # Python dependencies
│   ├── data/
│   │   └── ckd_imputed.csv   # Training dataset
│   ├── models/
│   │   └── ckd_final_model_full_data_gradient_boosting.pkl  # Trained ML model
│   └── venv/                  # Python virtual environment (not in git)
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Header.jsx           # Application header
│   │   │   ├── PredictionForm.jsx   # Input form component
│   │   │   ├── ResultsDashboard.jsx # Results visualization component
│   │   │   └── FormField.jsx        # Reusable form field component
│   │   ├── utils/
│   │   │   ├── api.js               # API communication utilities
│   │   │   └── predictionLogic.js   # Prediction result processing
│   │   ├── App.jsx                  # Main application component
│   │   ├── App.css                  # Application styles
│   │   └── main.jsx                 # Application entry point
│   ├── package.json                 # Node.js dependencies
│   └── vite.config.js              # Vite configuration
│
└── README.md                        # This file
```

## 🔌 API Documentation

### Endpoint: `/predict`

**Method**: `POST`

**Content-Type**: `application/json`

**Request Body**:
```json
{
  "age": 65,
  "bp": 140,
  "hemo": 12.5,
  "sg": 1,
  "sc": 1.2,
  "rbcc": 4.5,
  "pcv": 40,
  "htn": 1,
  "dm": 0
}
```

**Response**:
```json
{
  "class": 1,
  "confidence": 0.95,
  "prediction_text": "CKD",
  "patient_values": {
    "age": 65,
    "bp": 140,
    ...
  },
  "comparison_data": {
    "age": {
      "patient_value": 65,
      "ckd_mean": 60.5,
      "ckd_std": 12.3,
      "notckd_mean": 45.2,
      "notckd_std": 15.1,
      ...
    },
    ...
  }
}
```

**Field Descriptions**:
- `class`: Prediction class (0 = No CKD, 1 = CKD)
- `confidence`: Model confidence (0-1)
- `prediction_text`: Human-readable prediction
- `patient_values`: Original input values
- `comparison_data`: Statistical comparison with population data

## 🔧 Configuration

### Environment Variables

The frontend can be configured using environment variables:

- `VITE_API_URL`: Backend API URL (default: `/api`)

Create a `.env` file in the `frontend` directory:

```env
VITE_API_URL=http://localhost:5000
```

### Backend Configuration

The backend server runs on port 5000 by default. To change this, modify `app.py`:

```python
if __name__ == '__main__':
    app.run(debug=True, port=5000)  # Change port here
```

## 🧪 Testing

### Test the API

You can test the backend API using the provided test script:

```bash
cd backend
python test_api.py
```

Or use curl:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 65,
    "bp": 140,
    "hemo": 12.5,
    "sg": 1,
    "sc": 1.2,
    "rbcc": 4.5,
    "pcv": 40,
    "htn": 1,
    "dm": 0
  }'
```

## 🏗️ Building for Production

### Frontend Build

To create a production build of the frontend:

```bash
cd frontend
npm run build
```

The built files will be in the `frontend/dist` directory.

### Backend Deployment

For production deployment:

1. Set `debug=False` in `app.py`
2. Use a production WSGI server like Gunicorn:

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## ⚠️ Important Notes

- **Medical Disclaimer**: This tool is for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment.
- **Data Privacy**: Ensure patient data is handled according to applicable healthcare data protection regulations (e.g., HIPAA, GDPR).
- **Model Limitations**: The model is trained on specific datasets and may not generalize to all patient populations.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the MIT License.

## 👥 Authors

- Your Name/Organization

## 🙏 Acknowledgments

- Dataset and model training references
- Libraries and frameworks used
- Any other acknowledgments

## 📞 Support

For issues, questions, or contributions, please open an issue on the GitHub repository.

---

**Made with ❤️ for healthcare professionals**
