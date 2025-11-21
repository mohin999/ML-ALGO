from flask import Flask, request, jsonify
from flask_cors import CORS
from gradio_client import Client
import os

# Initialize app and enable CORS
app = Flask(__name__)
CORS(app)

# Connect to Hugging Face Space
client = Client("mohin999/test1")

# Home route
@app.route("/", methods=["GET"])
def home():
    return "🌸 ML Backend is running! Use /predict to get predictions."

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    
    sl = data.get("sepal_length")
    sw = data.get("sepal_width")
    pl = data.get("petal_length")
    pw = data.get("petal_width")
    
    # Call Hugging Face Space
    result = client.predict(
        sepal_length=sl,
        sepal_width=sw,
        petal_length=pl,
        petal_width=pw,
        api_name="/predict"
    )
    
    return jsonify({"prediction": result})

# Render requires dynamic PORT
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
