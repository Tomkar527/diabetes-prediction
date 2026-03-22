from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import pickle

# 1. Initialize the Flask app and enable CORS
app = Flask(__name__)
CORS(app) # This allows your HTML file to talk to this API

# 2. Load the trained machine learning model
try:
    with open("model2.pkl", "rb") as file:
        model = pickle.load(file)
except Exception as e:
    print(f"Error loading the model: {e}")
    model = None

# 3. Add a simple root endpoint to check if the API is running
@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Diabetes Prediction Flask API is running."})

# 4. Create the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict_diabetes():
    if model is None:
        return jsonify({"error": "Model is not loaded."}), 500

    try:
        # Get the JSON data sent from the website
        data = request.get_json()
        
        # Convert the dictionary into a pandas DataFrame (1 row)
        input_df = pd.DataFrame([data])
        
        # Make the prediction
        prediction = model.predict(input_df)
        
        # Extract the integer result (0 or 1)
        result = int(prediction[0])
        
        # Determine human-readable status
        status = "Diabetic" if result == 1 else "Non-Diabetic"
        
        # Send back the response
        return jsonify({
            "prediction": result,
            "status": status
        })
        
    except Exception as e:
        return jsonify({"error": f"Error making prediction: {str(e)}"}), 400

# 5. Run the server
if __name__ == '__main__':
    # I set the port to 8000 so it perfectly matches the fetch() URL in your index.html
    app.run(debug=True, port=8000)