# ===========================================================
# Flask Application for Regression Model Prediction
# -----------------------------------------------------------
# This web application:
# 1. Loads a trained regression model and scaler (both saved as .pkl files)
# 2. Serves a home page ('/')
# 3. Provides an API endpoint ('/predict_api') for JSON-based predictions
# ===========================================================

# ---------- Import Required Libraries ----------
import pickle                                  # For loading saved model and scaler
import numpy as np                             # For numerical array operations
from flask import Flask, request, jsonify, render_template  # For web framework

# ---------- Initialize Flask App ----------
app = Flask(__name__)  # Create Flask application instance

# ---------- Load Model and Scaler ----------
# Load your previously saved model and scaler
# Make sure 'regmodel.pkl' and 'scaler.pkl' are in the same directory
with open('regmodel.pkl', 'rb') as model_file:
    regmodel = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)


# ===========================================================
# ROUTE 1 — HOME PAGE
# -----------------------------------------------------------
# This route serves the HTML page 'home.html'
# Make sure you have a 'templates' folder with 'home.html' inside it
# ===========================================================
@app.route('/')
def home():
    """Render the home page."""
    return render_template('home.html')


# ===========================================================
# ROUTE 2 — PREDICTION API
# -----------------------------------------------------------
# This endpoint receives JSON input, scales it, and returns a prediction.
# Expected Input Format:
# {
#   "data": {
#       "feature1": value1,
#       "feature2": value2,
#       ...
#   }
# }
# -----------------------------------------------------------
# Example:
# curl -X POST http://127.0.0.1:5000/predict_api \
#      -H "Content-Type: application/json" \
#      -d '{"data": {"feature1": 3, "feature2": 1.2, "feature3": 0.5}}'
# ===========================================================
@app.route('/predict_api', methods=['POST'])
def predict_api():
    """
    API route for making predictions using JSON input.
    The input is transformed, scaled, and passed to the regression model.
    """

    # Extract JSON data from the POST request
    data = request.json['data']  # Get the 'data' part of the JSON body
    print("Received data:", data)

    # Convert the input data (dictionary values) into a NumPy array
    # Example: {"feature1": 10, "feature2": 5} -> [10, 5]
    feature_array = np.array(list(data.values())).reshape(1, -1)
    print("Feature array (before scaling):", feature_array)

    # Apply the same scaling used during model training
    new_data = scaler.transform(feature_array)
    print("Feature array (after scaling):", new_data)

    # Make prediction using the loaded regression model
    prediction = regmodel.predict(new_data)

    # Round the prediction for readability (e.g., 123.4567 -> 123.46)
    output = np.round(prediction, 2)
    print("Prediction result:", output[0])

    # Return prediction as a JSON response
    return jsonify(float(output[0]))


@app.route('/predict', methods=['POST'])
def predict():
    """ Route to handle submission from the HTML form"""
    # Extract form data from the POST request
    int_features = [float(x) for x in request.form.values()]
    final_feature = np.array(int_features).reshape(1, -1)
    input_data = scaler.transform(final_feature)
    print(input_data)
    input_pred = regmodel.predict(input_data)[0]
    print(input_pred)
    output = np.round(input_pred, 2)
    #print(output)
    return render_template('home.html', prediction_text = "The house predicted value is {}".format(output))



# ===========================================================
# MAIN ENTRY POINT
# -----------------------------------------------------------
# This ensures the app runs only when executed directly
# (and not when imported as a module).
# debug=True enables live reload during development
# ===========================================================
if __name__ == "__main__":
    app.run(debug=True)
