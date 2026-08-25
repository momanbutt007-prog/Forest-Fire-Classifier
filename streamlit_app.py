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
# CUSTOM STREAMLIT STYLING — Forest Green + Cream palette
# Python only - no external CSS file
# ============================================================

st.markdown(
    """
    <style>

        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&family=Inter:wght@400;500;600&display=swap');

        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        h1, h2, h3, h4 { font-family: 'Poppins', sans-serif !important; }

        /* ---------- App background: deep forest green ---------- */
        .stApp {
            background:
                radial-gradient(circle at 88% -5%, rgba(255, 130, 60, 0.08) 0%, transparent 38%),
                radial-gradient(circle at 8% 10%, rgba(214, 197, 150, 0.06) 0%, transparent 40%),
                radial-gradient(circle at 50% 105%, rgba(74, 143, 90, 0.10) 0%, transparent 55%),
                linear-gradient(160deg, #0a1510 0%, #0e1e15 45%, #0a140f 100%);
            color: #f2ead9;
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        /* ---------- Sidebar ---------- */
        [data-testid="stSidebar"] {
            background:
                radial-gradient(circle at 30% 0%, rgba(74,143,90,0.10), transparent 45%),
                radial-gradient(circle at 90% 40%, rgba(214,197,150,0.06), transparent 50%),
                linear-gradient(180deg, #0c1a13 0%, #08120d 100%);
            border-right: 1px solid rgba(214,197,150,0.10);
        }

        [data-testid="stSidebar"] * { color: #f2ead9 !important; }

        .sb-logo-wrap { text-align: center; padding: 0.6rem 0 0.4rem 0; }

        .sb-logo-badge {
            width: 60px;
            height: 60px;
            margin: 0 auto 0.55rem auto;
            border-radius: 18px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.85rem;
            background: linear-gradient(135deg, #1f5c39, #4a8f5a 55%, #d6c596);
            box-shadow: 0 10px 26px rgba(31,92,57,0.40);
        }

        .sb-title {
            font-size: 1.22rem;
            font-weight: 800;
            font-family: 'Poppins', sans-serif;
            margin-bottom: 0.12rem;
            color: #f2ead9;
        }

        .sb-subtitle {
            font-size: 0.7rem;
            color: rgba(242,234,217,0.55) !important;
            letter-spacing: 1.5px;
            text-transform: uppercase;
        }

        .sb-section-label {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-family: 'Poppins', sans-serif;
            font-weight: 700;
            font-size: 0.92rem;
            margin: 0.6rem 0 0.6rem 0;
            color: #f2ead9 !important;
        }

        .sb-section-label .icon-chip-sm {
            width: 25px;
            height: 25px;
            min-width: 25px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.82rem;
            background: linear-gradient(135deg, rgba(74,143,90,0.30), rgba(214,197,150,0.25));
            border: 1px solid rgba(242,234,217,0.14);
        }

        .sb-card {
            background: rgba(214,197,150,0.045);
            border: 1px solid rgba(214,197,150,0.14);
            border-radius: 14px;
            padding: 0.9rem 1rem;
            margin-bottom: 0.6rem;
            box-shadow: 0 6px 18px rgba(0,0,0,0.25);
        }

        .sb-card p {
            margin: 0.25rem 0;
            font-size: 0.86rem;
            color: rgba(242,234,217,0.85) !important;
        }

        .sb-card b { color: #d6c596 !important; }

        .sb-status-chip {
            display: flex;
            align-items: center;
            gap: 0.55rem;
            border-radius: 12px;
            padding: 0.6rem 0.85rem;
            margin-bottom: 0.5rem;
            font-size: 0.86rem;
            font-weight: 600;
        }

        .sb-status-online {
            background: rgba(74,143,90,0.14);
            border: 1px solid rgba(74,143,90,0.4);
            color: #8fd6a4 !important;
        }

        .sb-status-error {
            background: rgba(255,90,50,0.12);
            border: 1px solid rgba(255,90,50,0.35);
            color: #ffab8a !important;
        }

        .sb-class-pill {
            display: flex;
            align-items: center;
            gap: 0.55rem;
            background: rgba(214,197,150,0.04);
            border: 1px solid rgba(214,197,150,0.12);
            border-radius: 10px;
            padding: 0.5rem 0.75rem;
            margin-bottom: 0.4rem;
            font-size: 0.88rem;
        }

        /* ---------- Hero ---------- */
        .hero {
            padding: 35px 10px 25px 10px;
            text-align: center;
            border-radius: 22px;
            background: linear-gradient(135deg, rgba(31,92,57,0.35), rgba(214,197,150,0.08));
            border: 1px solid rgba(214,197,150,0.12);
            margin-bottom: 1.4rem;
            position: relative;
            overflow: hidden;
        }

        .hero::after {
            content: "";
            position: absolute;
            top: -50px;
            right: -50px;
            width: 200px;
            height: 200px;
            background: rgba(214,197,150,0.06);
            border-radius: 50%;
        }

        .hero-icon {
            font-size: 68px;
            margin-bottom: 5px;
        }

        .hero-title {
            font-size: 46px;
            font-weight: 800;
            letter-spacing: -1.5px;
            color: #f6f1e4;
            margin: 0;
        }

        .hero-title span {
            color: #ff8342;
        }

        .hero-subtitle {
            color: #cfc3a3;
            font-size: 16px;
            margin-top: 8px;
        }

        .status {
            display: inline-block;
            padding: 8px 18px;
            border-radius: 30px;
            background: rgba(74, 190, 100, 0.14);
            border: 1px solid rgba(74, 190, 100, 0.35);
            color: #8fe6ab;
            font-size: 13px;
            font-weight: 700;
            margin-top: 15px;
            letter-spacing: 0.5px;
        }

        /* ---------- Upload / result cards ---------- */
        .upload-card {
            padding: 25px;
            border-radius: 20px;
            background: rgba(214,197,150,0.035);
            border: 1px solid rgba(214,197,150,0.12);
        }

        .result-card {
            padding: 28px;
            border-radius: 22px;
            text-align: center;
            background: rgba(214,197,150,0.035);
            border: 1px solid rgba(214,197,150,0.12);
            margin-top: 15px;
        }

        .result-fire {
            border: 1px solid rgba(255,110,60,0.5);
            background: linear-gradient(160deg, rgba(255,90,40,0.12), rgba(214,197,150,0.03));
        }

        .result-safe {
            border: 1px solid rgba(74,190,100,0.45);
            background: linear-gradient(160deg, rgba(60,170,90,0.12), rgba(214,197,150,0.03));
        }

        .result-icon {
            font-size: 65px;
        }

        .result-title {
            font-size: 34px;
            font-weight: 800;
            color: #f6f1e4;
            margin-top: 5px;
        }

        .result-status {
            color: #d8cdb4;
            font-size: 16px;
        }

        /* ---------- Metric cards ---------- */
        .metric-card {
            padding: 18px;
            border-radius: 16px;
            background: rgba(214,197,150,0.035);
            border: 1px solid rgba(214,197,150,0.12);
            text-align: center;
            box-shadow: 0 8px 22px rgba(0,0,0,0.25);
        }

        .metric-label {
            color: #b7ab8d;
            font-size: 13px;
            letter-spacing: 0.5px;
        }

        .metric-value {
            color: #f6f1e4;
            font-size: 25px;
            font-weight: 750;
            margin-top: 4px;
            font-family: 'Poppins', sans-serif;
        }

        /* ---------- Native st.metric restyle ---------- */
        div[data-testid="stMetric"] {
            background: rgba(214,197,150,0.035);
            border: 1px solid rgba(214,197,150,0.12);
            border-radius: 16px;
            padding: 0.9rem 1rem 0.6rem 1rem;
            box-shadow: 0 8px 22px rgba(0,0,0,0.25);
        }

        div[data-testid="stMetricLabel"] { color: #cfc3a3 !important; }
        div[data-testid="stMetricValue"] { color: #f6f1e4 !important; font-family: 'Poppins', sans-serif; }

        /* ---------- Progress bar ---------- */
        div[data-testid="stProgress"] > div > div {
            background: linear-gradient(90deg, #4a8f5a, #d6c596) !important;
        }

        /* ---------- File uploader ---------- */
        div[data-testid="stFileUploaderDropzone"] {
            background:
                radial-gradient(circle at 20% 15%, rgba(74,143,90,0.10), transparent 55%),
                radial-gradient(circle at 80% 85%, rgba(214,197,150,0.08), transparent 55%),
                rgba(214,197,150,0.03);
            border: 2px dashed rgba(74,143,90,0.5);
            border-radius: 16px;
        }

        div[data-testid="stFileUploaderDropzone"]:hover {
            border-color: rgba(214,197,150,0.6);
        }

        div[data-testid="stFileUploaderDropzone"] button {
            background: linear-gradient(120deg, #1f5c39, #4a8f5a) !important;
            color: #f6f1e4 !important;
            border: none !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
        }

        /* ---------- Buttons ---------- */
        div.stButton > button {
            border-radius: 12px;
            font-weight: 700;
            background: linear-gradient(120deg, #1f5c39, #4a8f5a);
            color: #f6f1e4;
            border: none;
            box-shadow: 0 8px 20px rgba(31,92,57,0.35);
        }

        hr { border-color: rgba(214,197,150,0.14) !important; }

        .footer {
            text-align: center;
            color: #96a695;
            font-size: 13px;
            padding: 35px 0 10px;
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

    st.markdown(
        """
        <div class="sb-logo-wrap">
            <div class="sb-logo-badge">🔥</div>
            <div class="sb-title">Forest Fire AI</div>
            <div class="sb-subtitle">Wildfire Detection Engine</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown(
        '<div class="sb-section-label"><span class="icon-chip-sm">🤖</span> Model</div>',
        unsafe_allow_html=True,
    )

    if model_ready:
        st.markdown(
            '<div class="sb-status-chip sb-status-online">● Model loaded</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="sb-status-chip sb-status-error">● Model unavailable</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div class="sb-card">
            <p>🧠 <b>Architecture:</b> EfficientNetB2</p>
            <p>📐 <b>Input:</b> {IMG_SIZE} × {IMG_SIZE}</p>
            <p>🎯 <b>Output:</b> Binary classification</p>
            <p>⚖️ <b>Threshold:</b> {NO_FIRE_THRESHOLD:.2f}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown(
        '<div class="sb-section-label"><span class="icon-chip-sm">📊</span> Classes</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="sb-class-pill">🔥 Fire</div>
        <div class="sb-class-pill">🌲 No Fire</div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown(
        '<div class="sb-section-label"><span class="icon-chip-sm">⚙️</span> Runtime</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="sb-card">
            <p>⚡ Local inference</p>
            <p>🧩 TensorFlow / Keras</p>
        </div>
        """,
        unsafe_allow_html=True,
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