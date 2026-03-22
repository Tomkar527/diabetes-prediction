from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model
with open('model2.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    # Serves the glassmorphism webpage
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract JSON data from the frontend request
        data = request.json
        
        # Create a DataFrame with the exact column names expected by the pipeline
        input_data = pd.DataFrame([{
            'Pregnancies': data['Pregnancies'],
            'Glucose': data['Glucose'],
            'BloodPressure': data['BloodPressure'],
            'SkinThickness': data['SkinThickness'],
            'Insulin': data['Insulin'],
            'BMI': data['BMI'],
            'DiabetesPedigreeFunction': data['DiabetesPedigreeFunction'],
            'Age': data['Age']
        }])
        
        # Predict the outcome
        prediction = model.predict(input_data)
        
        # Return the prediction to the frontend
        return jsonify({'prediction': int(prediction[0])})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)