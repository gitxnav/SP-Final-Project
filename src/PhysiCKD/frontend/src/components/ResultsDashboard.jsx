import React from 'react';
import { Chart as ChartJS, ArcElement, Tooltip, Legend, CategoryScale, LinearScale, BarElement, Title, PointElement, LineElement } from 'chart.js';
import { Doughnut, Bar, Line } from 'react-chartjs-2';
import { getModelName } from '../utils/predictionLogic';

ChartJS.register(ArcElement, Tooltip, Legend, CategoryScale, LinearScale, BarElement, Title, PointElement, LineElement);

const ResultsDashboard = ({ prediction, formData }) => {
    if (!prediction) {
        return (
            <div id="resultsContent" className="results-content">
                <div className="empty-state">
                    <p>Fill out the form and select a model to generate predictions.</p>
                </div>
            </div>
        );
    }

    const { riskScore, riskLevel, model, confidence, prediction: predictionText, factors, comparison_data, patient_values } = prediction;
    const riskClass = `risk-${riskLevel}`;

    // Feature display names mapping
    const featureDisplayNames = {
        'age': 'Age (years)',
        'bp': 'Blood Pressure (mmHg)',
        'hemo': 'Hemoglobin (g/dL)',
        'sg': 'Specific Gravity',
        'sc': 'Serum Creatinine (mg/dL)',
        'rbcc': 'Red Blood Cell Count (million/μL)',
        'pcv': 'Packed Cell Volume (%)',
        'htn': 'Hypertension',
        'dm': 'Diabetes Mellitus'
    };

    // Risk Chart Data
    const riskChartData = {
        labels: ['CKD Risk', 'Low Risk'],
        datasets: [{
            data: [riskScore, 100 - riskScore],
            backgroundColor: [
                riskScore > 70 ? '#dc3545' : riskScore > 40 ? '#ffc107' : '#28a745',
                '#e0e0e0'
            ],
            borderWidth: 0
        }]
    };

    // Parameters Chart Data
    const normalizedData = {
        'Serum Creatinine': (factors.sc / 76) * 100,
        'Hemoglobin': (factors.hemo / 17.8) * 100,
        'Age': (factors.age / 90) * 100,
        'Blood Pressure': (factors.bp / 180) * 100
    };

    const paramsChartData = {
        labels: Object.keys(normalizedData),
        datasets: [{
            label: 'Normalized Values (%)',
            data: Object.values(normalizedData),
            backgroundColor: '#667eea',
            borderRadius: 6
        }]
    };

    const paramsOptions = {
        responsive: true,
        plugins: { legend: { display: false } }, // Hide legend
        scales: {
            y: {
                max: 100,
                ticks: { callback: (value) => value + '%' }
            }
        }
    };


    // Risk Factors Analysis
    const factorLabels = [];
    const factorValues = [];
    const factorColors = [];

    if (factors.age > 60) { factorLabels.push('Age > 60'); factorValues.push(15); factorColors.push('#dc3545'); }
    if (factors.bp > 140) { factorLabels.push('High BP'); factorValues.push(20); factorColors.push('#dc3545'); }
    if (factors.sc > 1.5) { factorLabels.push('High Creatinine'); factorValues.push(25); factorColors.push('#dc3545'); }
    if (formData.dm === 'yes') { factorLabels.push('Diabetes'); factorValues.push(15); factorColors.push('#ffc107'); }
    if (formData.htn === 'yes') { factorLabels.push('Hypertension'); factorValues.push(10); factorColors.push('#ffc107'); }
    if (factors.hemo < 10) { factorLabels.push('Low Hemoglobin'); factorValues.push(20); factorColors.push('#dc3545'); }

    if (factorLabels.length === 0) {
        factorLabels.push('No Major Risk Factors');
        factorValues.push(0);
        factorColors.push('#28a745');
    }

    const factorsChartData = {
        labels: factorLabels,
        datasets: [{
            label: 'Risk Contribution',
            data: factorValues,
            backgroundColor: factorColors,
            borderRadius: 6
        }]
    };

    const factorsOptions = {
        indexAxis: 'y',
        responsive: true,
        plugins: { legend: { display: false } },
    }

    // Comparison Chart Data - Patient vs CKD/Non-CKD averages
    let comparisonChartData = null;
    let comparisonChartOptions = null;

    // Categorical features that use mode instead of mean
    const categoricalFeatures = ['dm', 'htn', 'sg'];

    if (comparison_data && Object.keys(comparison_data).length > 0) {
        const features = Object.keys(comparison_data);
        const labels = features.map(f => featureDisplayNames[f] || f);
        
        const patientData = features.map(f => comparison_data[f].patient_value);
        // Use mode for categorical features, mean for numerical features
        const ckdMeanData = features.map(f => {
            const isCategorical = categoricalFeatures.includes(f);
            return isCategorical ? comparison_data[f].ckd_mode : comparison_data[f].ckd_mean;
        });
        const notckdMeanData = features.map(f => {
            const isCategorical = categoricalFeatures.includes(f);
            return isCategorical ? comparison_data[f].notckd_mode : comparison_data[f].notckd_mean;
        });
        const ckdStdData = features.map(f => comparison_data[f].ckd_std);
        const notckdStdData = features.map(f => comparison_data[f].notckd_std);

        comparisonChartData = {
            labels: labels,
            datasets: [
                {
                    label: 'Patient Value',
                    data: patientData,
                    backgroundColor: '#667eea',
                    borderColor: '#667eea',
                    borderWidth: 2,
                    borderRadius: 6,
                    order: 1
                },
                {
                    label: 'CKD Average/Mode',
                    data: ckdMeanData,
                    backgroundColor: '#dc3545',
                    borderColor: '#dc3545',
                    borderWidth: 2,
                    borderRadius: 6,
                    order: 2
                },
                {
                    label: 'Non-CKD Average/Mode',
                    data: notckdMeanData,
                    backgroundColor: '#28a745',
                    borderColor: '#28a745',
                    borderWidth: 2,
                    borderRadius: 6,
                    order: 3
                }
            ]
        };

        // Find max value for scaling
        const allValues = [...patientData, ...ckdMeanData, ...notckdMeanData];
        const maxValue = Math.max(...allValues) * 1.1;

        comparisonChartOptions = {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                },
                tooltip: {
                    callbacks: {
                        afterLabel: function(context) {
                            const datasetIndex = context.datasetIndex;
                            const dataIndex = context.dataIndex;
                            const feature = features[dataIndex];
                            const isCategorical = categoricalFeatures.includes(feature);
                            
                            if (datasetIndex === 1) {
                                // CKD dataset
                                if (isCategorical) {
                                    return 'Most common value';
                                } else {
                                    const std = ckdStdData[dataIndex];
                                    return std !== null ? `Std Dev: ±${std.toFixed(2)}` : '';
                                }
                            } else if (datasetIndex === 2) {
                                // Non-CKD dataset
                                if (isCategorical) {
                                    return 'Most common value';
                                } else {
                                    const std = notckdStdData[dataIndex];
                                    return std !== null ? `Std Dev: ±${std.toFixed(2)}` : '';
                                }
                            }
                            return '';
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: maxValue,
                    title: {
                        display: true,
                        text: 'Value'
                    }
                },
                x: {
                    ticks: {
                        maxRotation: 45,
                        minRotation: 45
                    }
                }
            }
        };
    }

    // Individual Feature Comparison Charts
    const individualComparisonCharts = [];
    if (comparison_data) {
        const keyFeatures = ['hemo', 'sc', 'age', 'bp', 'rbcc', 'pcv', 'sg', 'htn', 'dm'];
        keyFeatures.forEach(feature => {
            if (comparison_data[feature]) {
                const data = comparison_data[feature];
                const isCategorical = categoricalFeatures.includes(feature);
                
                // Use mode for categorical, mean for numerical
                const ckdValue = isCategorical ? data.ckd_mode : data.ckd_mean;
                const notckdValue = isCategorical ? data.notckd_mode : data.notckd_mean;
                
                const chartData = {
                    labels: ['Patient', 'CKD Average/Mode', 'Non-CKD Average/Mode'],
                    datasets: [{
                        label: featureDisplayNames[feature] || feature,
                        data: [
                            data.patient_value,
                            ckdValue,
                            notckdValue
                        ],
                        backgroundColor: [
                            '#667eea',
                            '#dc3545',
                            '#28a745'
                        ],
                        borderColor: [
                            '#667eea',
                            '#dc3545',
                            '#28a745'
                        ],
                        borderWidth: 2,
                        borderRadius: 6
                    }]
                };

                const chartOptions = {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            callbacks: {
                                afterLabel: function(context) {
                                    const index = context.dataIndex;
                                    if (isCategorical) {
                                        if (index === 1 || index === 2) {
                                            return 'Most common value';
                                        }
                                    } else {
                                        if (index === 1) {
                                            return data.ckd_std !== null ? `Std Dev: ±${data.ckd_std.toFixed(2)}` : '';
                                        } else if (index === 2) {
                                            return data.notckd_std !== null ? `Std Dev: ±${data.notckd_std.toFixed(2)}` : '';
                                        }
                                    }
                                    return '';
                                }
                            }
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: featureDisplayNames[feature] || feature
                            }
                        }
                    }
                };

                individualComparisonCharts.push({
                    feature: feature,
                    title: featureDisplayNames[feature] || feature,
                    data: chartData,
                    options: chartOptions
                });
            }
        });
    }

    return (
        <div id="resultsContent" className="results-content prediction-results">
            <div className="prediction-results">
                <div className="prediction-card">
                    <h3>Prediction Result</h3>
                    <div className="prediction-value">{riskScore}%</div>
                    <div className="prediction-label">CKD Risk Score</div>
                    <div className={`risk-indicator ${riskClass}`}>
                        {riskLevel.toUpperCase()} RISK
                    </div>
                    <div className="model-info">
                        Model: {getModelName(model)} | Confidence: {confidence}%
                    </div>
                </div>

                <div className="stats-grid">
                    <div className="stat-card">
                        <h4>Prediction</h4>
                        <div className="value">{predictionText}</div>
                    </div>
                    <div className="stat-card">
                        <h4>Risk Level</h4>
                        <div className="value">{riskLevel.toUpperCase()}</div>
                    </div>
                    <div className="stat-card">
                        <h4>Confidence</h4>
                        <div className="value">{confidence}%</div>
                    </div>
                    <div className="stat-card">
                        <h4>Model Used</h4>
                        <div className="value" style={{ fontSize: '1.2em' }}>{getModelName(model)}</div>
                    </div>
                </div>

                <div className="chart-container">
                    <h3>Risk Score Distribution</h3>
                    <div style={{ height: '300px', display: 'flex', justifyContent: 'center' }}>
                        <Doughnut data={riskChartData} options={{ maintainAspectRatio: false }} />
                    </div>
                </div>

                <div className="chart-container">
                    <h3>Key Parameters Comparison</h3>
                    <Bar data={paramsChartData} options={paramsOptions} />
                </div>

                <div className="chart-container">
                    <h3>Risk Factors Analysis</h3>
                    <Bar data={factorsChartData} options={factorsOptions} />
                </div>

                {comparisonChartData && (
                    <div className="chart-container">
                        <h3>Patient Values vs Population Averages</h3>
                        <p style={{ marginBottom: '20px', color: '#666', fontSize: '0.9em' }}>
                            Comparison of patient values against average values for CKD and Non-CKD populations
                        </p>
                        <div style={{ height: '400px' }}>
                            <Bar data={comparisonChartData} options={comparisonChartOptions} />
                        </div>
                    </div>
                )}

                {individualComparisonCharts.length > 0 && (
                    <div className="chart-container">
                        <h3>Detailed Feature Comparisons</h3>
                        <p style={{ marginBottom: '20px', color: '#666', fontSize: '0.9em' }}>
                            Individual comparison of key features with population averages
                        </p>
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '20px' }}>
                            {individualComparisonCharts.map((chart, index) => (
                                <div key={index} style={{ height: '300px' }}>
                                    <h4 style={{ marginBottom: '10px', fontSize: '1em' }}>{chart.title}</h4>
                                    <Bar data={chart.data} options={chart.options} />
                                </div>
                            ))}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

export default ResultsDashboard;
