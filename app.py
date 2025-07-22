# app.py

from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load model with error handling
try:
    model = pickle.load(open("model.pkl", "rb"))
    print("Model loaded successfully")
except FileNotFoundError:
    print("Error: model.pkl not found. Please run model_train.py first.")
    model = None
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Try to load feature names if available
try:
    feature_names = pickle.load(open("feature_names.pkl", "rb"))
    print(f"Feature names loaded: {feature_names}")
except:
    # Default feature names for heart failure dataset
    feature_names = [
        'age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 
        'ejection_fraction', 'high_blood_pressure', 'platelets',
        'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time'
    ]
    print(f"Using default feature names: {feature_names}")

@app.route('/')
def home():
    return render_template("index.html", features=feature_names)

@app.route('/predict', methods=["POST"])
def predict():
    if request.method == "POST":
        try:
            if model is None:
                return render_template("index.html", 
                                     prediction_text="Error: Model not loaded. Please check model.pkl file.",
                                     features=feature_names)
            
            # Extract data from form in the correct order
            data = []
            for feature in feature_names:
                value = request.form.get(feature)
                if value is None or value == '':
                    return render_template("index.html", 
                                         prediction_text=f"Error: Missing value for {feature}",
                                         features=feature_names)
                try:
                    data.append(float(value))
                except ValueError:
                    return render_template("index.html", 
                                         prediction_text=f"Error: Invalid value for {feature}. Please enter a number.",
                                         features=feature_names)
            
            # Make prediction
            input_array = np.array([data])
            prediction = model.predict(input_array)
            prediction_proba = model.predict_proba(input_array)
            
            # Get confidence score
            confidence = max(prediction_proba[0]) * 100
            
            # Generate output message
            if prediction[0] == 1:
                output = f"⚠ High Risk: Patient has elevated risk of mortality (Confidence: {confidence:.1f}%)"
                risk_class = "high-risk"
            else:
                output = f"✅ Lower Risk: Patient has lower risk of mortality (Confidence: {confidence:.1f}%)"
                risk_class = "low-risk"
            
            return render_template("index.html", 
                                 prediction_text=output,
                                 risk_class=risk_class,
                                 confidence=confidence,
                                 features=feature_names)
            
        except Exception as e:
            return render_template("index.html", 
                                 prediction_text=f"Error during prediction: {str(e)}",
                                 features=feature_names)

@app.route('/info')
def info():
    """Route to display information about the features"""
    feature_info = {
        'age': 'Age of the patient (years)',
        'anaemia': 'Decrease of red blood cells or hemoglobin (0 = No, 1 = Yes)',
        'creatinine_phosphokinase': 'Level of CPK enzyme in blood (mcg/L)',
        'diabetes': 'If patient has diabetes (0 = No, 1 = Yes)',
        'ejection_fraction': 'Percentage of blood leaving heart at each contraction (%)',
        'high_blood_pressure': 'If patient has hypertension (0 = No, 1 = Yes)',
        'platelets': 'Platelets in blood (kiloplatelets/mL)',
        'serum_creatinine': 'Level of serum creatinine in blood (mg/dL)',
        'serum_sodium': 'Level of serum sodium in blood (mEq/L)',
        'sex': 'Woman or man (0 = Woman, 1 = Man)',
        'smoking': 'If patient smokes (0 = No, 1 = Yes)',
        'time': 'Follow-up period (days)'
    }
    return render_template("info.html", feature_info=feature_info)

@app.errorhandler(404)
def not_found(error):
    return render_template("error.html", error="Page not found"), 404

@app.errorhandler(500)
def server_error(error):
    return render_template("error.html", error="Internal server error"), 500

if __name__ == "__main__":
    if not os.path.exists('templates'):
        print("Warning: 'templates' directory not found. Creating basic template structure.")
        os.makedirs('templates', exist_ok=True)
        
    app.run(debug=True, host='0.0.0.0', port=5000)
