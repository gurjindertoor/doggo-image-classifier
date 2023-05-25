import tensorflow as tf
import tensorflow_hub as hub
import os
import numpy as np
import pandas as pd
import requests
from flask import Flask, render_template, send_from_directory, url_for
from flask_uploads import UploadSet, IMAGES, configure_uploads
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField
from PIL import Image

# Define image size
IMG_SIZE = 224

explanations = {
    "good with children": "1 implies poor while 5 implies excellent",
    "good with other dogs": "1 implies poor while 5 implies excellent",
    "shedding": "1 implies minimal while 5 implies maximal",
    "grooming": "1 implies minimally needed while 5 implies maximally needed",
    "drooling": "1 implies minimum while 5 implies maximal",
    "coat length": "1 implies short while 5 implies long",
    "good with strangers": "1 implies poor while 5 implies excellent",
    "playfulness": "1 implies minimal while 5 implies maximal",
    "protectiveness": "1 implies minimal while 5 implies maximal",
    "trainability": "1 implies hard while 5 implies easy",
    "energy": "1 implies little while 5 implies a lot",
    "barking": "1 implies minimal while 5 implies maximal",
    "min life expectancy": "Measured in years",
    "max life expectancy": "Measured in years",
    "max height male": "Measured in inches",
    "max height female": "Measured in inches",
    "max weight male": "Measured in lbs.",
    "max weight female": "Measured in lbs.",
    "min height male": "Measured in inches",
    "min height female": "Measured in inches",
    "min weight male": "Measured in lbs.",
    "min weight female": "Measured in lbs.",
}

# 1. Create a function for preprocessing images
def process_image(image_path, img_size=IMG_SIZE):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, size=[IMG_SIZE, IMG_SIZE])
    return image

# Load the model
def load_model(model_path):
    print(f"Loading saved model from: {model_path}")
    model = tf.keras.models.load_model(
        model_path, custom_objects={"KerasLayer": hub.KerasLayer}
    )
    return model

# Predict labels for custom images
def predict_custom_images(model, custom_image_paths):
    custom_images = [process_image(image_path) for image_path in custom_image_paths]
    custom_data = tf.data.Dataset.from_tensor_slices(custom_images).batch(32)
    custom_preds = model.predict(custom_data)
    custom_pred_labels = [
        get_pred_label(custom_preds[i]) for i in range(len(custom_preds))
    ]
    return custom_pred_labels, custom_images

# Define labels and unique breeds
labels_csv = pd.read_csv("doggo_files\labels.csv")
labels = labels_csv["breed"].to_numpy()
unique_breeds = np.unique(labels)

# Turn prediction probabilities into their respective label
def get_pred_label(prediction_probabilities):
    label = unique_breeds[np.argmax(prediction_probabilities)]
    label_words = label.replace("_", " ").split()
    capitalized_label = " ".join([word.capitalize() for word in label_words])
    return capitalized_label

app = Flask(__name__)
app.config["SECRET_KEY"] = "doggo_classifier"
app.config["UPLOADED_PHOTOS_DEST"] = "uploads"

photos = UploadSet("photos", IMAGES)
configure_uploads(app, photos)


class UploadForm(FlaskForm):
    photo = FileField(
        validators=[
            FileAllowed(photos, "Only images are allowed"),
            FileRequired("File field should not be empty"),
        ]
    )
    submit = SubmitField("Upload")

@app.route("/uploads/<filename>")
def get_file(filename):
    return send_from_directory(app.config["UPLOADED_PHOTOS_DEST"], filename)

@app.route("/", methods=["GET", "POST"])
def upload_image():
    form = UploadForm()
    if form.validate_on_submit():
        filename = photos.save(form.photo.data)
        file_path = os.path.join(app.config["UPLOADED_PHOTOS_DEST"], filename)
        file_url = url_for("get_file", filename=filename)

        # Convert the uploaded image to JPG format
        jpg_path = os.path.splitext(file_path)[0] + ".jpg"
        img = Image.open(file_path)
        img.save(jpg_path, "JPEG")

        custom_pred_labels, custom_images = predict_custom_images(
            loaded_full_model, [jpg_path]
        )

        return render_template(
            "index.html", form=form, file_url=file_url, pred_labels=custom_pred_labels
        )
    else:
        file_url = None
        return render_template("index.html", form=form, file_url=file_url)

@app.route("/breed_info/<breed>")
def get_breed_info(breed):
    api_url = "https://api.api-ninjas.com/v1/dogs?name={}".format(breed)
    response = requests.get(
        api_url, headers={"X-Api-Key": "J5ESuUp63zg/LdInrS0IZQ==6ZiDcXST8vmet06p"}
    )
    if response.status_code == requests.codes.ok:
        breed_info = response.json()
        if isinstance(breed_info, list) and breed_info:
            breed_info = breed_info[0]
        return render_template(
            "breed_info.html",
            breed_info=breed_info,
            breed=breed,
            explanations=explanations,
        )
    else:
        error_message = "Error: {} - {}".format(response.status_code, response.text)
        return render_template(
            "breed_info.html", breed=breed, error_message=error_message
        )

if __name__ == "__main__":
    loaded_full_model = load_model("model\doggo-classifier-eff-net.h5")
    app.run(port=3000, debug=True)
