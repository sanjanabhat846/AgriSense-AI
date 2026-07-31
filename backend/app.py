from flask import Flask, render_template, request, jsonify
from pathlib import Path

from services.predictor import predict_disease

# ============================================================
# Flask App
# ============================================================

app = Flask(__name__)

# ============================================================
# Upload Folder
# ============================================================

UPLOAD_FOLDER = Path("backend/uploads")
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)

# ============================================================
# Home Route
# ============================================================

@app.route("/")
def home():
    return render_template("index.html")


# ============================================================
# About Route
# ============================================================

@app.route("/about")
def about():

    return jsonify({

        "project": "AgriSense AI",

        "model": "EfficientNetB0",

        "dataset": "PlantVillage",

        "classes": 38,

        "accuracy": "95.87%"

    })


# ============================================================
# Prediction Route
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:

        return jsonify({

            "success": False,

            "message": "No image uploaded."

        }), 400

    file = request.files["image"]

    if file.filename == "":

        return jsonify({

            "success": False,

            "message": "Please choose an image."

        }), 400

    image_path = UPLOAD_FOLDER / file.filename

    file.save(image_path)

    try:

        result = predict_disease(image_path)

        print(result)


        if result.get("success") is False:
            return jsonify(result), 400

        return jsonify(result)  

    except Exception as e:

        return jsonify({

            "success": False,

            "message": str(e)

        }), 500


# ============================================================
# Run Server
# ============================================================

if __name__ == "__main__":

    app.run(debug=True)