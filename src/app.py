"""
Streamlit Application for CKD Detection Project
Interactive dashboard for data exploration, model training and prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import json
from io import StringIO

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent))

from step01_data_loading import DataLoader
from step02_data_processing import DataProcessor
from step03_feature_engineering import FeatureEngineer
from step04_model_training import ModelTrainer
from step05_model_prediction import ModelPredictor

# Page configuration
st.set_page_config(
    page_title="CKD Detection Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables"""
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'data_processed' not in st.session_state:
        st.session_state.data_processed = False
    if 'features_engineered' not in st.session_state:
        st.session_state.features_engineered = False
    if 'models_trained' not in st.session_state:
        st.session_state.models_trained = False
    if 'df' not in st.session_state:
        st.session_state.df = None


def load_data_section():
    """Data loading section"""
    st.header("📥 Data Loading")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        data_path = st.text_input(
            "Data Path", 
            value="data/raw/chronic_kindey_disease.csv",
            help="Path to the raw CSV file"
        )
    
    with col2:
        st.write("")
        st.write("")
        if st.button("Load Data", type="primary", use_container_width=True):
            with st.spinner("Loading data..."):
                try:
                    loader = DataLoader(data_path)
                    df = loader.load_data()
                    
                    if loader.validate_schema():
                        st.session_state.df = df
                        st.session_state.data_loaded = True
                        st.success(f"✅ Data loaded successfully! Shape: {df.shape}")
                    else:
                        st.error("❌ Schema validation failed!")
                        
                except Exception as e:
                    st.error(f"❌ Error loading data: {str(e)}")
    
    if st.session_state.data_loaded:
        st.divider()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", st.session_state.df.shape[0])
        with col2:
            st.metric("Total Features", st.session_state.df.shape[1])
        with col3:
            numeric_cols = st.session_state.df.select_dtypes(include=[np.number]).shape[1]
            st.metric("Numeric Features", numeric_cols)
        with col4:
            categorical_cols = st.session_state.df.select_dtypes(include=['object']).shape[1]
            st.metric("Categorical Features", categorical_cols)
        
        # Show data preview
        with st.expander("📊 Data Preview (TOP 20)", expanded=True):
            st.dataframe(st.session_state.df.head(20), use_container_width=True)

        # Show data types
        with st.expander("📋 Data Types"):
            dtype_df = pd.DataFrame({
                'Column': st.session_state.df.dtypes.index,
                'Data Type': st.session_state.df.dtypes.values,
                'Non-Null Count': st.session_state.df.count().values,
                'Null Count': st.session_state.df.isnull().sum().values
            })
            st.dataframe(dtype_df, use_container_width=True)


        st.divider()
        col1 = st.columns(1)

        # Show data informations
        with st.expander("ℹ️ Data Information", expanded=True):
            datainfo_path = "data/raw/chronic_kindey_disease_info.txt"
            uploaded_file = st.file_uploader(datainfo_path,)
            if uploaded_file is not None:
                stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                st.write(stringio.read())

def process_data_section():
    """Data processing section"""
    st.header("⚙️ Data Processing")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first!")
        return
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        if st.button("Process Data", type="primary", use_container_width=True):
            with st.spinner("Processing data..."):
                try:
                    processor = DataProcessor()
                    processor.df = st.session_state.df.copy()
                    processor.clean_data()
                    st.session_state.df = processor.df
                    st.session_state.data_processed = True
                    st.success("✅ Data processed successfully!")
                except Exception as e:
                    st.error(f"❌ Error processing data: {str(e)}")
    
    if st.session_state.data_processed:
        st.divider()
        st.subheader("Processing Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Target Variable Distribution**")
            target_counts = st.session_state.df['status'].value_counts()
            st.bar_chart(target_counts)
        
        with col2:
            st.write("**Target Statistics**")
            for status, count in target_counts.items():
                percentage = (count / len(st.session_state.df)) * 100
                st.metric(label=f"{status.upper()}", value=count, delta=f"{percentage:.1f}%")

def feature_engineering_section():
    """Feature engineering section"""
    st.header("🔧 Feature Engineering")
    
    if not st.session_state.data_processed:
        st.warning("⚠️ Please process data first!")
        return
        
    
    st.info("This step will create two datasets: **Imputed** (for Gradient Boosting) and **Normalized** (for KNN/SVM)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Create Imputed Dataset", use_container_width=True):
            with st.spinner("Creating imputed dataset..."):
                try:
                    engineer = FeatureEngineer()
                    df_imputed = engineer.process_pipeline_imputed()
                    st.success("✅ Imputed dataset created!")
                    st.dataframe(df_imputed.head(), use_container_width=True)
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    with col2:
        if st.button("Create Normalized Dataset", use_container_width=True):
            with st.spinner("Creating normalized dataset..."):
                try:
                    engineer = FeatureEngineer()
                    df_normalized = engineer.process_pipeline_normalized()
                    st.success("✅ Normalized dataset created!")
                    st.dataframe(df_normalized.head(), use_container_width=True)
                    st.session_state.features_engineered = True
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    # Show feature statistics if available
    stats_path = Path('reports/feature_engineering_stats.json')
    if stats_path.exists():
        with st.expander("📊 Feature Engineering Statistics"):
            with open(stats_path, 'r') as f:
                stats = json.load(f)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Features", stats.get('total_features', 'N/A'))
            with col2:
                st.metric("Missing Values", stats.get('missing_values', 'N/A'))
            with col3:
                st.metric("Data Shape", str(stats.get('data_shape', 'N/A')))
            
            st.write("**Selected Features:**")
            st.write(", ".join(stats.get('feature_list', [])))

def model_training_section():
    """Model training section"""
    st.header("🤖 Model Training")
    
    if not st.session_state.features_engineered:
        st.warning("⚠️ Please complete feature engineering first!")
        return
    
    st.write("Train multiple models: KNN, SVM, Gradient Boosting, and Histogram Gradient Boosting")
    
    tab1, tab2, tab3 = st.tabs(["Training Model", "Features importance", "Model Evaluation"])
    
    with tab1:
        st.subheader("Training Models")

    # Model selection
    models_to_train = st.multiselect(
        "Select models to train:",
        ["KNN", "SVM", "GradientBoosting", "HistGradientBoosting"],
        default=["KNN", "SVM"]
        )
    
    if st.button("Train Selected Models", type="primary", use_container_width=True):
        if not models_to_train:
            st.warning("Please select at least one model!")
            return
        
        trainer = ModelTrainer()
        results = {}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_models = len(models_to_train)
        
        for idx, model_name in enumerate(models_to_train):
            status_text.text(f"Training {model_name}... ({idx+1}/{total_models})")
            
            try:
                if model_name in ["KNN", "SVM"]:
                    X_train, X_test, y_train, y_test = trainer.load_data('data/processed/ckd_normalized.csv')
                else:
                    X_train, X_test, y_train, y_test = trainer.load_data('data/processed/ckd_imputed.csv')
                
                if model_name == "KNN":
                    result = trainer.train_knn(X_train, X_test, y_train, y_test)
                elif model_name == "SVM":
                    result = trainer.train_svm(X_train, X_test, y_train, y_test)
                elif model_name == "GradientBoosting":
                    result = trainer.train_gradient_boosting_imputed(X_train, X_test, y_train, y_test)
                elif model_name == "HistGradientBoosting":
                    result = trainer.train_hist_gradient_boosting(X_train, X_test, y_train, y_test)
                
                results[model_name] = result
                st.success(f"✅ {model_name} trained! F1-Score: {result['f1_score']:.4f}")
                
            except Exception as e:
                st.error(f"❌ Error training {model_name}: {str(e)}")
            
            progress_bar.progress((idx + 1) / total_models)
        
        status_text.text("Training complete!")
        st.session_state.models_trained = True

        with tab2:
            st.subheader("Feature Importance")

            if results.get('GradientBoosting'):
                st.write("**Gradient Boosting Feature Importance**")
                fig, ax = plt.subplots(figsize=(10, 6))
                feature_importances = results['GradientBoosting']['feature_importances']
                features = results['GradientBoosting']['feature_names']
                sorted_idx = np.argsort(feature_importances)[::-1]
                ax.barh(np.array(features)[sorted_idx], np.array(feature_importances)[sorted_idx], color='teal')
                ax.set_xlabel('Importance')
                ax.set_title('Feature Importance - Gradient Boosting')
                st.pyplot(fig)
                plt.close()
            else:
                st.info("Gradient Boosting model not trained, cannot show feature importance.")
            
            if results.get('HistGradientBoosting'):
                st.write("**Histogram Gradient Boosting Feature Importance**")
                fig, ax = plt.subplots(figsize=(10, 6))
                feature_importances = results['HistGradientBoosting']['feature_importances']
                features = results['HistGradientBoosting']['feature_names']
                sorted_idx = np.argsort(feature_importances)[::-1]
                ax.barh(np.array(features)[sorted_idx], np.array(feature_importances)[sorted_idx], color='orange')
                ax.set_xlabel('Importance')
                ax.set_title('Feature Importance - Histogram Gradient Boosting')
                st.pyplot(fig)
                plt.close() 
            else:
                st.info("Histogram Gradient Boosting model not trained, cannot show feature importance.")   

            st.write("**Saved Feature Importance Plots**")

            if results.get('KNN') or results.get('SVM'):
                st.write("KNN and SVM do not provide feature importance directly.")
            
            feature_importance_path = Path('figures/models/feature_importance.png')
            if feature_importance_path.exists():
                st.image(str(feature_importance_path), caption="Feature Importance", use_column_width=True)
            else:
                st.info("Feature importance plot not available.")
        
        with tab3:
            st.subheader("Model Evaluation")

            # Display results
            if results:
                st.subheader("Training Results")
                
                # Create comparison dataframe
                comparison_df = pd.DataFrame({
                    'Model': list(results.keys()),
                    'Accuracy': [r['accuracy'] for r in results.values()],
                    'Precision': [r['precision'] for r in results.values()],
                    'Recall': [r['recall'] for r in results.values()],
                    'F1-Score': [r['f1_score'] for r in results.values()]
                })
                
                st.dataframe(
                    comparison_df.style.highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1-Score']),
                    use_container_width=True
                )
                
                # Show confusion matrices
                st.subheader("Confusion Matrices")
                cols = st.columns(len(results))
                
                for idx, (model_name, result) in enumerate(results.items()):
                    with cols[idx]:
                        cm_path = Path(f'figures/models/{model_name.lower()}_confusion_matrix.png')
                        if cm_path.exists():
                            st.image(str(cm_path), caption=model_name)
            else:
                st.info("No training results to display.")

def prediction_section():
    """Model prediction section"""
    st.header("🔮 Make Predictions")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train models first!")
        return
    
    # Initialize predictor
    try:
        predictor = ModelPredictor()
        predictor.load_all_models()
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        return
    
    st.subheader("Enter Patient Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        hemo = st.number_input("Hemoglobin (g/dL)", min_value=0.0, max_value=20.0, value=15.4, step=0.1)
        sg = st.number_input("Specific Gravity", min_value=1.000, max_value=1.030, value=1.020, step=0.005, format="%.3f")
        sc = st.number_input("Serum Creatinine (mg/dL)", min_value=0.0, max_value=20.0, value=1.2, step=0.1)
    
    with col2:
        rbcc = st.number_input("Red Blood Cell Count (millions/cmm)", min_value=0.0, max_value=10.0, value=5.2, step=0.1)
        pcv = st.number_input("Packed Cell Volume (%)", min_value=0.0, max_value=60.0, value=44.0, step=1.0)
        htn = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    
    with col3:
        dm = st.selectbox("Diabetes Mellitus", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        bp = st.number_input("Blood Pressure (mmHg)", min_value=0.0, max_value=200.0, value=80.0, step=1.0)
        age = st.number_input("Age (years)", min_value=0, max_value=120, value=48, step=1)
    
    patient_data = {
        'hemo': hemo, 'sg': sg, 'sc': sc, 'rbcc': rbcc, 'pcv': pcv,
        'htn': float(htn), 'dm': float(dm), 'bp': bp, 'age': float(age)
    }
    
    if st.button("Predict with All Models", type="primary", use_container_width=True):
        with st.spinner("Making predictions..."):
            try:
                results = predictor.predict_with_all_models(patient_data)
                
                st.divider()
                st.subheader("Prediction Results")
                
                # Individual model predictions
                cols = st.columns(len(predictor.models))
                
                for idx, (model_name, result) in enumerate(results.items()):
                    if model_name != 'consensus' and 'prediction' in result:
                        with cols[idx]:
                            prediction = result['prediction'].upper()
                            color = "🔴" if prediction == "CKD" else "🟢"
                            st.markdown(f"### {color} {model_name}")
                            st.markdown(f"**{prediction}**")
                            
                            if result['probability']:
                                prob = result['probability'][result['prediction']]
                                st.progress(prob)
                                st.caption(f"Confidence: {prob:.1%}")
                
                # Consensus
                if 'consensus' in results:
                    st.divider()
                    st.subheader("🎯 Consensus Prediction")
                    
                    consensus = results['consensus']['prediction'].upper()
                    confidence = results['consensus']['confidence']
                    agreement = results['consensus']['agreement']
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if consensus == "CKD":
                            st.error(f"### 🔴 {consensus}")
                        else:
                            st.success(f"### 🟢 {consensus}")
                    
                    with col2:
                        st.metric("Agreement", agreement)
                    
                    with col3:
                        st.metric("Confidence", f"{confidence:.1%}")
                
            except Exception as e:
                st.error(f"❌ Error making prediction: {str(e)}")

def analysis_section():
    """Data analysis section"""
    st.header("📊 Data Analysis")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first!")
        return
    
    tab1, tab2, tab3 = st.tabs(["Missing Values", "Statistical Summary", "Distributions"])
    
    with tab1:
        st.subheader("Missing Values Analysis")
        
        missing_count = st.session_state.df.isnull().sum()
        missing_percent = (st.session_state.df.isnull().sum() / len(st.session_state.df)) * 100
        
        missing_df = pd.DataFrame({
            'Feature': missing_count.index,
            'Missing Count': missing_count.values,
            'Missing %': missing_percent.values
        })
        missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
        
        if len(missing_df) > 0:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(missing_df['Feature'], missing_df['Missing %'], color='coral')
                ax.set_xlabel('Missing Percentage (%)')
                ax.set_title('Missing Values by Feature')
                ax.grid(axis='x', alpha=0.3)
                st.pyplot(fig)
                plt.close()
            
            with col2:
                st.dataframe(
                    missing_df.style.background_gradient(subset=['Missing %'], cmap='Reds'),
                    use_container_width=True,
                    hide_index=True
                )
        else:
            st.success("✅ No missing values found!")
    
    with tab2:
        st.subheader("Statistical Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Numerical Features**")
            numeric_summary = st.session_state.df.describe()
            st.dataframe(numeric_summary, use_container_width=True)
        
        with col2:
            st.write("**Categorical Features**")
            categorical_summary = st.session_state.df.describe(include=['object'])
            st.dataframe(categorical_summary, use_container_width=True)
    
    with tab3:
        st.subheader("Feature Distributions")
        
        numeric_cols = st.session_state.df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            selected_feature = st.selectbox("Select Feature", numeric_cols)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Histogram**")
                fig, ax = plt.subplots(figsize=(8, 4))
                st.session_state.df[selected_feature].dropna().hist(bins=30, ax=ax, color='skyblue', edgecolor='black')
                ax.set_xlabel(selected_feature)
                ax.set_ylabel('Frequency')
                ax.set_title(f'Distribution of {selected_feature}')
                st.pyplot(fig)
                plt.close()
            
            with col2:
                st.write("**Box Plot**")
                fig, ax = plt.subplots(figsize=(8, 4))
                st.session_state.df.boxplot(column=selected_feature, ax=ax)
                ax.set_ylabel(selected_feature)
                ax.set_title(f'Box Plot of {selected_feature}')
                st.pyplot(fig)
                plt.close()

def sidebar():
    """Sidebar configuration"""
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/kidney.png", width=100)
        st.title("CKD Detection")
        st.markdown("---")
        
        st.subheader("Navigation")
        page = st.radio(
            "Go to",
            ["Data Loading", "Data Processing", "Data Analysis", "Feature Engineering", "Model Training", "Predictions"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        st.subheader("Pipeline Status")
        status_items = [
            ("Data Loaded", st.session_state.data_loaded),
            ("Data Processed", st.session_state.data_processed),
            ("Features Engineered", st.session_state.features_engineered),
            ("Models Trained", st.session_state.models_trained)
        ]
        
        for label, status in status_items:
            if status:
                st.success(f"✅ {label}")
            else:
                st.info(f"⏳ {label}")
        
        st.markdown("---")
        
        st.subheader("About")
        st.info(
            "**CKD Detection Dashboard**\n\n"
            "End-to-end ML pipeline for Chronic Kidney Disease detection.\n\n"
            "Built with Streamlit 🎈"
        )
        
        return page


def main():
    """Main application"""
    init_session_state()
    
    st.markdown('<h1 class="main-header">🏥 Chronic Kidney Disease Detection Dashboard</h1>', unsafe_allow_html=True)
    
    page = sidebar()
    
    st.markdown("---")
    
    if page == "Data Loading":
        load_data_section()
    elif page == "Data Processing":
        process_data_section()
    elif page == "Feature Engineering":
        feature_engineering_section()
    elif page == "Model Training":
        model_training_section()
    elif page == "Predictions":
        prediction_section()
    elif page == "Data Analysis":
        analysis_section()


if __name__ == "__main__":
    main()