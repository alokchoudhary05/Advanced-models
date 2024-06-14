from flask import Flask, render_template, request
import pickle
import joblib
import warnings
from dashboard import dashboard_bp


app = Flask(__name__)
app.register_blueprint(dashboard_bp, url_prefix='/dashboard')



# Load models
pregnancy_model = joblib.load(open("models/pregnancy_model.pkl", "rb"))
heart_model = pickle.load(open("models/Heart.sav", 'rb'))
diabetic_model = pickle.load(open("models/Diabetes.sav", 'rb'))

@app.route('/')
def home():
    return render_template('base.html')

@app.route('/pregnancy', methods=['GET', 'POST'])
def pregnancy():
    risk_level = None
    color = None
    if request.method == 'POST':
        age = request.form['age']
        diastolicBP = request.form['diastolicBP']
        BS = request.form['BS']
        bodyTemp = request.form['bodyTemp']
        heartRate = request.form['heartRate']
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            predicted_risk = pregnancy_model.predict([[age, diastolicBP, BS, bodyTemp, heartRate]])
        
        if predicted_risk[0] == 0:
            risk_level = "Low Risk"
            color = "green"
        elif predicted_risk[0] == 1:
            risk_level = "Medium Risk"
            color = "orange"
        else:
            risk_level = "High Risk"
            color = "red"
    
    return render_template('pregnancy.html', risk_level=risk_level, color=color)




@app.route('/heart', methods=['GET', 'POST'])
def heart():
    prediction_text = None
    if request.method == 'POST':
        # Convert 'sex' and 'cp' inputs to the required integer values
        sex = 0 if request.form['sex'] == 'Male' else 1
        cp_dict = {
            'Low pain': 0,
            'Mild pain': 1,
            'Moderate pain': 2,
            'Extreme pain': 3
        }
        cp = cp_dict[request.form['cp']]

        data = [
            request.form['age'],
            sex,
            cp,
            request.form['trestbps'],
            request.form['chol'],
            request.form['fbs'],
            request.form['restecg'],
            request.form['thalach'],
            request.form['exang'],
            request.form['oldpeak'],
            request.form['slope'],
            request.form['ca'],
            request.form['thal']
        ]

        input_data = [data]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            heart_prediction = heart_model.predict(input_data)
        
        if heart_prediction[0] == 1:
            prediction_text = 'The person is having heart disease'
        else:
            prediction_text = 'The person does not have any heart disease'
    
    return render_template('heart.html', prediction_text=prediction_text)





@app.route('/diabetes', methods=['GET', 'POST'])
def diabetes():
    prediction_text = None
    if request.method == 'POST':
        data = [
            request.form['Pregnancies'],
            request.form['Glucose'],
            request.form['BloodPressure'],
            request.form['SkinThickness'],
            request.form['Insulin'],
            request.form['BMI'],
            request.form['DiabetesPedigreeFunction'],
            request.form['Age']
        ]

        input_data = [data]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prediction = diabetic_model.predict(input_data)

        if prediction[0] == 1:
            prediction_text = 'The person is diabetic'
        else:
            prediction_text = 'The person is not diabetic'
    
    return render_template('diabetes.html', prediction_text=prediction_text)





if __name__ == '__main__':
    app.run(debug=True)
