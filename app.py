from flask import Flask, request, jsonify
from gradio_client import Client

app = Flask(__name__)
CORS(app)
client = Client("mohin999/test1")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    result = client.predict(
        sepal_length=data["sepal_length"],
        sepal_width=data["sepal_width"],
        petal_length=data["petal_length"],
        petal_width=data["petal_width"],
        api_name="/predict"
    )
    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
