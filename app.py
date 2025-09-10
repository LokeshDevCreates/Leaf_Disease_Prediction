from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image
import json

# Load metrics once
with open("metrics.json", "r") as f:
    METRICS = json.load(f)
app = Flask(__name__)

# Load your trained model
MODEL_PATH = "plant_disease_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# Class labels (update based on your training classes)
CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Corn___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn___Common_rust",
    "Corn___Northern_Leaf_Blight",
    "Corn___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

# Function to prettify class names
def prettify_classname(classname):
    return classname.replace("___", " ").replace("_", " ")

# Upload folder
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def index():
    prediction_text = None
    filename = None
    confidence = None

    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", prediction="No file uploaded")
        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", prediction="No file selected")

        if file:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(file.filename))
            file.save(filepath)

            # Preprocess image
            img = image.load_img(filepath, target_size=(224, 224))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            preds = model.predict(img_array)
            class_index = np.argmax(preds)
            class_name = CLASS_NAMES[class_index]
            pretty_name = prettify_classname(class_name)
            confidence = float(np.max(preds))

            # Healthy vs Diseased
            if "healthy" in class_name.lower():
                prediction_text = "✅ There is no disease in the leaf."
            else:
                prediction_text = f"⚠️ The leaf has a disease: {pretty_name}"

            filename = file.filename

    return render_template(
        "index.html",
        prediction=prediction_text,
        filename=filename,
        confidence=confidence,
        accuracy=METRICS["accuracy"],
        loss=METRICS["loss"]
    )

if __name__ == "__main__":
    app.run(debug=True)
