from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Forest Fire AI",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "forest_fire_deployment.keras"

IMG_SIZE = 260
NO_FIRE_THRESHOLD = 0.70


# ============================================================
# CUSTOM STREAMLIT STYLING
# Python only - no external CSS file
# ============================================================

st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(
                    circle at top right,
                    rgba(255, 100, 20, 0.12),
                    transparent 30%
                ),
                linear-gradient(
                    135deg,
                    #0b0f0d 0%,
                    #111814 50%,
                    #0a0d0b 100%
                );
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        [data-testid="stSidebar"] {
            background: #0d130f;
            border-right: 1px solid rgba(255,255,255,0.08);
        }

        .hero {
            padding: 35px 10px 20px 10px;
            text-align: center;
        }

        .hero-icon {
            font-size: 70px;
            margin-bottom: 5px;
        }

        .hero-title {
            font-size: 48px;
            font-weight: 800;
            letter-spacing: -2px;
            color: white;
            margin: 0;
        }

        .hero-title span {
            color: #ff6422;
        }

        .hero-subtitle {
            color: #aab4ad;
            font-size: 17px;
            margin-top: 8px;
        }

        .status {
            display: inline-block;
            padding: 8px 16px;
            border-radius: 30px;
            background: rgba(40, 190, 100, 0.12);
            border: 1px solid rgba(40, 190, 100, 0.3);
            color: #6ee7a1;
            font-size: 13px;
            font-weight: 700;
            margin-top: 15px;
        }

        .upload-card {
            padding: 25px;
            border-radius: 20px;
            background: rgba(255,255,255,0.035);
            border: 1px solid rgba(255,255,255,0.08);
        }

        .result-card {
            padding: 28px;
            border-radius: 22px;
            text-align: center;
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            margin-top: 15px;
        }

        .result-fire {
            border: 1px solid rgba(255,80,40,0.45);
            background: rgba(255,70,30,0.08);
        }

        .result-safe {
            border: 1px solid rgba(70,200,120,0.4);
            background: rgba(50,180,100,0.07);
        }

        .result-icon {
            font-size: 65px;
        }

        .result-title {
            font-size: 34px;
            font-weight: 800;
            color: white;
            margin-top: 5px;
        }

        .result-status {
            color: #b8c2bc;
            font-size: 16px;
        }

        .metric-card {
            padding: 18px;
            border-radius: 16px;
            background: rgba(255,255,255,0.035);
            border: 1px solid rgba(255,255,255,0.07);
            text-align: center;
        }

        .metric-label {
            color: #8f9b94;
            font-size: 13px;
        }

        .metric-value {
            color: white;
            font-size: 25px;
            font-weight: 750;
            margin-top: 4px;
        }

        .footer {
            text-align: center;
            color: #68736c;
            font-size: 13px;
            padding: 35px 0 10px;
        }

        div.stButton > button {
            border-radius: 12px;
            font-weight: 700;
        }

        div[data-testid="stFileUploader"] {
            border-radius: 16px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-icon">🔥</div>
        <div class="hero-title">
            Forest <span>Fire AI</span>
        </div>
        <div class="hero-subtitle">
            Intelligent forest fire detection using deep learning
        </div>
        <div class="status">
            ● AI MODEL ONLINE
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MODEL
# ============================================================

@st.cache_resource
def load_fire_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )


try:
    model = load_fire_model()
    model_ready = True
except Exception as exc:
    model = None
    model_ready = False
    st.error(f"Could not load the AI model: {exc}")


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## 🔥 Forest Fire AI")

    st.markdown("---")

    st.markdown("### 🤖 Model")

    if model_ready:
        st.success("Model loaded")
    else:
        st.error("Model unavailable")

    st.markdown(
        f"""
        **Architecture:** EfficientNetB2  
        
        **Input:** {IMG_SIZE} × {IMG_SIZE}  
        
        **Output:** Binary classification  
        
        **Threshold:** {NO_FIRE_THRESHOLD:.2f}
        """
    )

    st.markdown("---")

    st.markdown("### 📊 Classes")

    st.markdown(
        """
        🔥 **Fire**  
        🌲 **No Fire**
        """
    )

    st.markdown("---")

    st.caption(
        "Local inference • TensorFlow / Keras"
    )


# ============================================================
# MAIN CONTENT
# ============================================================

left, right = st.columns([1.05, 0.95], gap="large")


# ============================================================
# UPLOAD SECTION
# ============================================================

with left:

    st.markdown("### 📷 Upload Forest Image")

    st.markdown(
        """
        Upload a forest image and let the AI model
        analyze whether fire is present.
        """
    )

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "webp"],
        help="Supported formats: JPG, JPEG, PNG and WEBP",
    )

    if uploaded_file is not None:

        try:
            image = Image.open(uploaded_file).convert("RGB")

            st.image(
                image,
                caption="Uploaded image",
                use_container_width=True,
            )

            st.caption(
                f"Original size: {image.width} × {image.height}"
            )

        except (UnidentifiedImageError, OSError):
            st.error("The uploaded file is not a valid image.")
            image = None

    else:
        image = None

        st.info(
            "👆 Upload a forest image to start the analysis."
        )


# ============================================================
# PREDICTION SECTION
# ============================================================

with right:

    st.markdown("### 🧠 AI Analysis")

    if image is None:

        st.markdown(
            """
            <div class="result-card">
                <div class="result-icon">🌲</div>
                <div class="result-title">
                    Ready to Analyze
                </div>
                <div class="result-status">
                    Upload an image to begin detection.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    elif not model_ready:

        st.error(
            "The model could not be loaded. "
            "Check the models folder."
        )

    else:

        analyze = st.button(
            "🔥 Analyze Image",
            use_container_width=True,
            type="primary",
        )

        if analyze:

            with st.spinner("Analyzing image with AI..."):

                resized = image.resize(
                    (IMG_SIZE, IMG_SIZE)
                )

                image_array = np.asarray(
                    resized,
                    dtype=np.float32,
                )

                image_array = np.expand_dims(
                    image_array,
                    axis=0,
                )

                prediction = model.predict(
                    image_array,
                    verbose=0,
                )

                no_fire_probability = float(
                    prediction[0][0]
                )

                fire_probability = (
                    1.0 - no_fire_probability
                )

            if no_fire_probability >= NO_FIRE_THRESHOLD:

                label = "NO FIRE"
                icon = "🌲"
                status = "No fire detected"
                result_class = "result-safe"
                confidence = no_fire_probability

            else:

                label = "FIRE DETECTED"
                icon = "🔥"
                status = "Potential forest fire detected"
                result_class = "result-fire"
                confidence = fire_probability

            st.markdown(
                f"""
                <div class="result-card {result_class}">
                    <div class="result-icon">{icon}</div>
                    <div class="result-title">
                        {label}
                    </div>
                    <div class="result-status">
                        {status}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("")

            m1, m2 = st.columns(2)

            with m1:
                st.metric(
                    "🔥 Fire Probability",
                    f"{fire_probability * 100:.2f}%",
                )

            with m2:
                st.metric(
                    "🌲 No Fire Probability",
                    f"{no_fire_probability * 100:.2f}%",
                )

            st.progress(
                min(max(confidence, 0.0), 1.0),
                text=f"Confidence: {confidence * 100:.2f}%",
            )

            if fire_probability > 0.5:
                st.warning(
                    "⚠️ The model indicates a possible fire. "
                    "Verify the result with additional evidence "
                    "before taking action."
                )
            else:
                st.success(
                    "🌲 The model indicates no fire in this image."
                )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.markdown("---")

st.markdown("### 📊 Model Information")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-label">MODEL</div>
            <div class="metric-value">EfficientNetB2</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-label">INPUT</div>
            <div class="metric-value">260 × 260</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-label">CLASSES</div>
            <div class="metric-value">2</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c4:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-label">THRESHOLD</div>
            <div class="metric-value">70%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Forest Fire AI • Deep Learning Computer Vision
        <br>
        TensorFlow / Keras • Local Model Inference
    </div>
    """,
    unsafe_allow_html=True,
)