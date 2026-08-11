import os
import uuid
from pathlib import Path

import numpy as np
import tensorflow as tf

from PIL import Image, UnidentifiedImageError
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    send_from_directory,
)
from werkzeug.utils import secure_filename


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# USE THE NEW DEPLOYMENT MODEL
MODEL_PATH = BASE_DIR / "models" / "forest_fire_deployment.keras"

UPLOAD_FOLDER = BASE_DIR / "uploads"

# IMPORTANT:
# Your model was trained using 260 x 260 images.
IMG_SIZE = 260

# Optimized threshold from your testing
NO_FIRE_THRESHOLD = 0.70

ALLOWED_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg",
    "webp"
}


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

app.config["SECRET_KEY"] = os.environ.get(
    "FLASK_SECRET_KEY",
    "forest-fire-secret-key"
)

app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)

# Maximum upload size = 10 MB
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


# Create uploads folder automatically
UPLOAD_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# MODEL VARIABLES
# ============================================================

model = None
model_error = None


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    global model
    global model_error

    print("\n" + "=" * 60)
    print("🔥 FOREST FIRE MODEL LOADING")
    print("=" * 60)

    print(f"Model path:")
    print(MODEL_PATH)

    # Check whether model exists
    if not MODEL_PATH.exists():

        model_error = (
            f"Model not found:\n{MODEL_PATH}\n\n"
            "Make sure forest_fire_deployment.keras "
            "is inside the models folder."
        )

        print("❌", model_error)

        return

    try:

        print("Loading Keras model...")

        # compile=False is correct for inference/deployment
        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False
        )

        model_error = None

        print("✅ Model loaded successfully!")
        print("Model input shape:", model.input_shape)
        print("Model output shape:", model.output_shape)

        print("=" * 60)

    except Exception as exc:

        model = None

        model_error = (
            f"Could not load Keras model:\n{exc}"
        )

        print("❌ MODEL LOADING ERROR")
        print(exc)

        print("=" * 60)


# Load model once when Flask starts
load_model()


# ============================================================
# FILE VALIDATION
# ============================================================

def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_image(image_path):

    if model is None:

        raise RuntimeError(
            model_error or
            "Model is not loaded."
        )

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = Image.open(image_path)

    # Convert to RGB
    image = image.convert("RGB")

    # Resize exactly like training
    image = image.resize(
        (IMG_SIZE, IMG_SIZE)
    )

    # --------------------------------------------------------
    # Convert image to NumPy
    # --------------------------------------------------------

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    prediction = model.predict(
        image_array,
        verbose=0
    )

    # Your model outputs sigmoid probability
    no_fire_probability = float(
        prediction[0][0]
    )

    fire_probability = (
        1.0 - no_fire_probability
    )

    # --------------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------------

    if no_fire_probability >= NO_FIRE_THRESHOLD:

        label = "No Fire"

        confidence = no_fire_probability

        status = "No fire detected"

        icon = "🌲"

        result_class = "safe"

    else:

        label = "Fire"

        confidence = fire_probability

        status = "Fire detected"

        icon = "🔥"

        result_class = "danger"

    # --------------------------------------------------------
    # RETURN RESULTS
    # --------------------------------------------------------

    return {

        "label": label,

        "confidence": confidence * 100,

        "fire_probability":
            fire_probability * 100,

        "no_fire_probability":
            no_fire_probability * 100,

        "status": status,

        "icon": icon,

        "result_class": result_class,
    }


# ============================================================
# HOME PAGE
# ============================================================

@app.route(
    "/",
    methods=["GET", "POST"]
)
def index():

    result = None

    image_url = None

    # --------------------------------------------------------
    # POST REQUEST
    # --------------------------------------------------------

    if request.method == "POST":

        # Check model
        if model is None:

            flash(
                model_error or
                "The model is not loaded.",
                "error"
            )

            return render_template(
                "index.html",
                result=None,
                image_url=None,
                model_ready=False,
                threshold=NO_FIRE_THRESHOLD
            )

        # ----------------------------------------------------
        # Check uploaded file
        # ----------------------------------------------------

        if "image" not in request.files:

            flash(
                "Please choose an image first.",
                "error"
            )

            return render_template(
                "index.html",
                result=None,
                image_url=None,
                model_ready=True,
                threshold=NO_FIRE_THRESHOLD
            )

        file = request.files["image"]

        # Empty filename
        if not file or file.filename == "":

            flash(
                "Please choose an image first.",
                "error"
            )

            return render_template(
                "index.html",
                result=None,
                image_url=None,
                model_ready=True,
                threshold=NO_FIRE_THRESHOLD
            )

        # ----------------------------------------------------
        # Check extension
        # ----------------------------------------------------

        if not allowed_file(file.filename):

            flash(
                "Allowed formats: JPG, JPEG, PNG, WEBP.",
                "error"
            )

            return render_template(
                "index.html",
                result=None,
                image_url=None,
                model_ready=True,
                threshold=NO_FIRE_THRESHOLD
            )

        # ----------------------------------------------------
        # Generate safe unique filename
        # ----------------------------------------------------

        original_filename = secure_filename(
            file.filename
        )

        filename = (
            f"{uuid.uuid4().hex}_"
            f"{original_filename}"
        )

        file_path = (
            UPLOAD_FOLDER /
            filename
        )

        file.save(file_path)

        # ----------------------------------------------------
        # Validate image
        # ----------------------------------------------------

        try:

            with Image.open(file_path) as img:

                img.verify()

            # ------------------------------------------------
            # Make prediction
            # ------------------------------------------------

            result = predict_image(
                file_path
            )

            image_url = url_for(
                "uploaded_file",
                filename=filename
            )

        # Invalid image
        except (
            UnidentifiedImageError,
            OSError
        ):

            file_path.unlink(
                missing_ok=True
            )

            flash(
                "The uploaded file is not a valid image.",
                "error"
            )

        # Prediction error
        except Exception as exc:

            file_path.unlink(
                missing_ok=True
            )

            flash(
                f"Prediction failed: {exc}",
                "error"
            )

    # --------------------------------------------------------
    # Render page
    # --------------------------------------------------------

    return render_template(
        "index.html",

        result=result,

        image_url=image_url,

        threshold=NO_FIRE_THRESHOLD,

        model_ready=model is not None,
    )


# ============================================================
# SERVE UPLOADED IMAGES
# ============================================================

@app.route(
    "/uploads/<filename>"
)
def uploaded_file(filename):

    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        filename
    )


# ============================================================
# FILE TOO LARGE
# ============================================================

@app.errorhandler(413)
def too_large(_error):

    flash(
        "Image is too large. Maximum size is 10 MB.",
        "error"
    )

    return redirect(
        url_for("index")
    )


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":

    print("\n🔥 Forest Fire Detection System")
    print("Model:", MODEL_PATH)

    if model is not None:

        print("✅ Model status: READY")

    else:

        print("❌ Model status: NOT READY")
        print(model_error)

    print("\n🌐 Starting Flask...")
    print("http://127.0.0.1:5000\n")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )