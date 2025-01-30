from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Charger le fichier local random_forest.pkl
try:
    model = joblib.load("random_forest.pkl")
    print("Model loaded successfully from random_forest.pkl")
except Exception as e:
    print(f"Could not load model: {e}")
    model = None

@app.route("/", methods=["GET"])
def index():
    return {"message": "Hello, MLOps Demo !"}

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        data = request.get_json()  # Ex. {"features": [5.1, 3.5, 1.4, 0.2]}
        features = data["features"]
        prediction = model.predict([features])
        return jsonify({"prediction": float(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
