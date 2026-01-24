
export function generateMockPrediction(data) {
    // Mock prediction logic based on input values
    let riskScore = 0;
    let riskLevel = 'low';
    
    // Calculate risk based on various factors
    const age = parseFloat(data.age) || 50;
    const bp = parseFloat(data.bp) || 120;
    const sc = parseFloat(data.sc) || 1.0;
    const bu = parseFloat(data.bu) || 20;
    const hemo = parseFloat(data.hemo) || 12;
    const bgr = parseFloat(data.bgr) || 100;
    
    // Age factor
    if (age > 60) riskScore += 15;
    else if (age > 50) riskScore += 10;
    
    // Blood pressure
    if (bp > 140) riskScore += 20;
    else if (bp > 130) riskScore += 10;
    
    // Serum creatinine (high is bad)
    if (sc > 1.5) riskScore += 25;
    else if (sc > 1.2) riskScore += 15;
    
    // Blood urea
    if (bu > 40) riskScore += 15;
    else if (bu > 30) riskScore += 8;
    
    // Hemoglobin (low is bad)
    if (hemo < 10) riskScore += 20;
    else if (hemo < 12) riskScore += 10;
    
    // Blood glucose
    if (bgr > 200) riskScore += 15;
    else if (bgr > 140) riskScore += 8;
    
    // Diabetes
    if (data.dm === 'yes') riskScore += 15;
    
    // Hypertension
    if (data.htn === 'yes') riskScore += 10;
    
    // Anemia
    if (data.ane === 'yes') riskScore += 12;
    
    // Abnormal RBC
    if (data.rbc === 'abnormal') riskScore += 10;
    
    // Poor appetite
    if (data.appet === 'poor') riskScore += 8;
    
    // Normalize to percentage (0-100)
    riskScore = Math.min(100, Math.max(0, riskScore));
    
    // Determine risk level
    if (riskScore >= 70) riskLevel = 'high';
    else if (riskScore >= 40) riskLevel = 'medium';
    else riskLevel = 'low';
    
    // Model-specific adjustments (mock)
    const modelMultipliers = {
        'random_forest': 1.0,
        'svm': 0.95,
        'extreme_gradient': 1.05,
        'nns': 1.02
    };
    
    const adjustedScore = riskScore * (modelMultipliers[data.model] || 1.0);
    const finalScore = Math.min(100, Math.max(0, adjustedScore));
    
    return {
        riskScore: finalScore.toFixed(1),
        riskLevel: riskLevel,
        model: data.model,
        prediction: finalScore > 50 ? 'CKD Positive' : 'CKD Negative',
        confidence: (100 - Math.abs(finalScore - 50) * 2).toFixed(1),
        factors: {
            age: age,
            bp: bp,
            sc: sc,
            bu: bu,
            hemo: hemo,
            bgr: bgr
        }
    };
}

export function getModelName(modelKey) {
    const modelNames = {
        'random_forest': 'Random Forest',
        'svm': 'SVM (Support Vector Machine)',
        'extreme_gradient': 'Extreme Gradient Boosting',
        'nns': 'Neural Networks'
    };
    return modelNames[modelKey] || modelKey;
}
