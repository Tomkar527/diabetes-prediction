from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd

# 1. DEFINE THE APP FIRST (This fixes your error)
app = Flask(__name__)
CORS(app)

# 2. LOAD YOUR MODEL
with open('model2.pkl', 'rb') as f:
    model = pickle.load(f)

# 3. DEFINE YOUR ROUTES
@app.route('/')
def home():
    # Flask looks for this file inside the 'templates' folder
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])
        prediction = model.predict(df)
        
        result = {
            'prediction': int(prediction[0]),
            'outcome': 'Diabetic' if int(prediction[0]) == 1 else 'Non-Diabetic'
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 4. RUN THE APP
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)