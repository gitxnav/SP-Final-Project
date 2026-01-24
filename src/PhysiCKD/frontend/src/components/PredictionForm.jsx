import React, { useState, useEffect } from 'react';
import FormField from './FormField';
import { predictCKD } from '../utils/api';

const sectionFields = {
    patientInfo: ['age', 'bp', 'htn', 'dm'],
    labResults: ['hemo', 'sg', 'sc', 'rbcc', 'pcv']
};

const initialFormData = {
    age: '', bp: '', htn: '', dm: '',
    hemo: '', sg: '', sc: '', rbcc: '', pcv: ''
};

const PredictionForm = ({ onSubmit, onClear }) => {
    const [formData, setFormData] = useState(initialFormData);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [fieldErrors, setFieldErrors] = useState({});

    const validateField = (name, value) => {
        const numValue = parseFloat(value);
        const intValue = parseInt(value);

        switch (name) {
            case 'age':
                if (!value || value.trim() === '') return 'Age is required';
                if (isNaN(intValue) || intValue < 0 || intValue > 120) {
                    return 'Age must be between 0 and 120 years';
                }
                return '';
            case 'bp':
                if (!value || value.trim() === '') return 'Blood Pressure is required';
                // Medically reasonable range: 40-250 mmHg (covers severe hypotension to hypertensive crisis)
                if (isNaN(numValue) || numValue < 40 || numValue > 250) {
                    return 'Blood Pressure must be between 40 and 250 mmHg';
                }
                return '';
            case 'htn':
                if (!value || value === '') return 'Hypertension selection is required';
                if (value !== 'yes' && value !== 'no') {
                    return 'Please select Yes or No';
                }
                return '';
            case 'dm':
                if (!value || value === '') return 'Diabetes Mellitus selection is required';
                if (value !== 'yes' && value !== 'no') {
                    return 'Please select Yes or No';
                }
                return '';
            case 'hemo':
                if (!value || value.trim() === '') return 'Hemoglobin is required';
                // Medically reasonable range: 3.0-20.0 g/dL (covers severe anemia to polycythemia)
                if (isNaN(numValue) || numValue < 3.0 || numValue > 20.0) {
                    return 'Hemoglobin must be between 3.0 and 20.0 g/dL';
                }
                return '';
            case 'pcv':
                if (!value || value.trim() === '') return 'Packed Cell Volume is required';
                // Medically reasonable range: 10-60% (covers severe anemia to polycythemia)
                if (isNaN(intValue) || intValue < 10 || intValue > 60) {
                    return 'Packed Cell Volume must be between 10 and 60%';
                }
                return '';
            case 'rbcc':
                if (!value || value.trim() === '') return 'Red Blood Cell Count is required';
                // Medically reasonable range: 1.0-8.0 million/μL (covers severe anemia to polycythemia)
                if (isNaN(numValue) || numValue < 1.0 || numValue > 8.0) {
                    return 'Red Blood Cell Count must be between 1.0 and 8.0 million/μL';
                }
                return '';
            case 'sc':
                if (!value || value.trim() === '') return 'Serum Creatinine is required';
                // Medically reasonable range: 0.2-30.0 mg/dL (covers normal to end-stage renal disease)
                if (isNaN(numValue) || numValue < 0.2 || numValue > 30.0) {
                    return 'Serum Creatinine must be between 0.2 and 30.0 mg/dL';
                }
                return '';
            case 'sg':
                if (!value || value === '') return 'Specific Gravity is required';
                // Valid specific gravity values: 1.005, 1.010, 1.015, 1.020, 1.025
                if (!['1', '2', '3', '4', '5'].includes(value)) {
                    return 'Please select a valid Specific Gravity value (1.005-1.025)';
                }
                return '';
            default:
                return '';
        }
    };

    const validateForm = () => {
        const errors = {};
        let isValid = true;

        Object.keys(formData).forEach(field => {
            const error = validateField(field, formData[field]);
            if (error) {
                errors[field] = error;
                isValid = false;
            }
        });

        setFieldErrors(errors);
        return isValid;
    };

    const handleChange = (e) => {
        const { name, value } = e.target;
        setFormData(prev => ({ ...prev, [name]: value }));
        
        // Clear error for this field when user starts typing
        if (fieldErrors[name]) {
            const error = validateField(name, value);
            setFieldErrors(prev => {
                const newErrors = { ...prev };
                if (error) {
                    newErrors[name] = error;
                } else {
                    delete newErrors[name];
                }
                return newErrors;
            });
        }
    };

    const isSectionComplete = (section) => {
        const fields = sectionFields[section];
        return fields.every(field => formData[field] && formData[field].toString().trim() !== '');
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError(null);

        // Validate all fields before submission
        if (!validateForm()) {
            setError('Please correct the errors in the form before submitting.');
            return;
        }

        setLoading(true);

        try {
            const apiData = {
                age: parseFloat(formData.age),
                bp: parseFloat(formData.bp),
                hemo: parseFloat(formData.hemo),
                pcv: parseFloat(formData.pcv),
                rbcc: parseFloat(formData.rbcc),
                sc: parseFloat(formData.sc),
                sg: parseInt(formData.sg),
                dm: formData.dm === 'yes' ? 1 : 0,
                htn: formData.htn === 'yes' ? 1 : 0,
            };

            const result = await predictCKD(apiData);

            // Transform result
            // Confidence is probability of the predicted class
            const prob = result.confidence;
            const isCKD = result.class === 1;
            // Calculate a "Risk Score" (0-100) where 100 is high certainty of CKD
            // If CKD (1), score is prob * 100
            // If No CKD (0), score is (1 - prob) * 100 ?? 
            // Actually, let's just use the probability of class 1 as risk score.
            // If class 1, prob(1) = prob.
            // If class 0, prob(0) = prob => prob(1) = 1 - prob.
            const riskScoreRaw = isCKD ? prob : (1 - prob);
            const riskScore = riskScoreRaw * 100;

            const transformedResult = {
                riskScore: riskScore.toFixed(1),
                riskLevel: riskScore > 70 ? 'high' : (riskScore > 40 ? 'medium' : 'low'),
                model: 'Gradient Boosting',
                prediction: isCKD ? 'CKD Positive' : 'CKD Negative',
                confidence: (prob * 100).toFixed(1),
                factors: apiData,
                comparison_data: result.comparison_data || null,
                patient_values: result.patient_values || apiData
            };

            onSubmit(transformedResult, formData);
        } catch (err) {
            console.error(err);
            setError(err.message || "Failed to connect to prediction service.");
        } finally {
            setLoading(false);
        }
    };

    const handleClear = () => {
        setFormData(initialFormData);
        setFieldErrors({});
        setError(null);
        onClear();
    };

    const renderIndicator = (section) => {
        const completed = isSectionComplete(section);
        return (
            <span className={`completion-indicator ${completed ? 'completed' : ''}`}>
                {completed ? '✓' : '○'}
            </span>
        );
    };

    return (
        <section className="form-section">
            <div className="form-container">
                <form id="predictionForm" onSubmit={handleSubmit} noValidate>

                    {/* Patient Information */}
                    <div className="form-group" data-section="patientInfo">
                        <div className="section-header">
                            <h2>Patient Information</h2>
                            {renderIndicator('patientInfo')}
                        </div>
                        <div className="form-row">
                            <FormField 
                                label="Age (years) [0-120]" 
                                id="age" 
                                name="age" 
                                type="number" 
                                min="0" 
                                max="120" 
                                step="1" 
                                value={formData.age} 
                                onChange={handleChange}
                                required={true}
                                error={fieldErrors.age}
                            />
                            <FormField 
                                label="Blood Pressure (mmHg) [40-250]" 
                                id="bp" 
                                name="bp" 
                                type="number" 
                                min="40" 
                                max="250" 
                                step="1" 
                                value={formData.bp} 
                                onChange={handleChange}
                                required={true}
                                error={fieldErrors.bp}
                            />
                        </div>
                        <div className="form-row">
                            <FormField 
                                label="Diabetes Mellitus" 
                                id="dm" 
                                name="dm" 
                                type="select" 
                                value={formData.dm} 
                                onChange={handleChange}
                                required={true}
                                error={fieldErrors.dm}
                                options={[{ value: 'yes', label: 'Yes' }, { value: 'no', label: 'No' }]}
                            />
                            <FormField 
                                label="Hypertension" 
                                id="htn" 
                                name="htn" 
                                type="select" 
                                value={formData.htn} 
                                onChange={handleChange}
                                required={true}
                                error={fieldErrors.htn}
                                options={[{ value: 'yes', label: 'Yes' }, { value: 'no', label: 'No' }]}
                            />
                        </div>
                    </div>

                    {/* Laboratory Results */}
                    <div className="form-group" data-section="labResults">
                        <div className="section-header">
                            <h2>Laboratory Results</h2>
                            {renderIndicator('labResults')}
                        </div>
                        <div className="form-row">
                            <FormField 
                                label="Hemoglobin (g/dL) [3.0-20.0]" 
                                id="hemo" 
                                name="hemo" 
                                type="number" 
                                min="3.0" 
                                max="20.0" 
                                step="0.1" 
                                value={formData.hemo} 
                                onChange={handleChange}
                                required={true}
                                error={fieldErrors.hemo}
                            />
                            <FormField 
                                label="Packed Cell Volume (%) [10-60]" 
                                id="pcv" 
                                name="pcv" 
                                type="number" 
                                min="10" 
                                max="60" 
                                step="1" 
                                value={formData.pcv} 
                                onChange={handleChange}
                                required={true}
                                error={fieldErrors.pcv}
                            />
                        </div>
                        <div className="form-row">
                            <FormField 
                                label="Red Blood Cell Count (million/μL) [1.0-8.0]" 
                                id="rbcc" 
                                name="rbcc" 
                                type="number" 
                                min="1.0" 
                                max="8.0" 
                                step="0.1" 
                                value={formData.rbcc} 
                                onChange={handleChange}
                                required={true}
                                error={fieldErrors.rbcc}
                            />
                            <FormField 
                                label="Serum Creatinine (mg/dL) [0.2-30.0]" 
                                id="sc" 
                                name="sc" 
                                type="number" 
                                min="0.2" 
                                max="30.0" 
                                step="0.1" 
                                value={formData.sc} 
                                onChange={handleChange}
                                required={true}
                                error={fieldErrors.sc}
                            />
                        </div>
                        <div className="form-row">
                            <FormField 
                                label="Specific Gravity" 
                                id="sg" 
                                name="sg" 
                                type="select" 
                                value={formData.sg} 
                                onChange={handleChange}
                                required={true}
                                error={fieldErrors.sg}
                                options={[
                                    { value: '1', label: '1.005' },
                                    { value: '2', label: '1.010' },
                                    { value: '3', label: '1.015' },
                                    { value: '4', label: '1.020' },
                                    { value: '5', label: '1.025' },
                                ]}
                            />
                        </div>
                    </div>

                    <div className="form-actions">
                        <button type="submit" className="btn-primary" disabled={loading}>
                            {loading ? 'Processing...' : 'Generate Prediction'}
                        </button>
                        <button type="button" className="btn-secondary" onClick={handleClear}>Clear Form</button>
                    </div>
                    {error && <div className="error-message" style={{ color: 'red', marginTop: '10px' }}>{error}</div>}
                </form>
            </div>
        </section>
    );
};

export default PredictionForm;
