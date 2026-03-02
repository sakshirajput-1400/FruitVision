import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import json

# ---------------- CONFIG ----------------
IMG_SIZE = 224
MODEL_ACCURACY = 94.0

st.set_page_config(page_title="FruitVision", layout="wide")

# ---------------- STRONG VISIBILITY FIX ----------------
st.markdown("""
<style>

/* App Background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #f4f7fb, #ffffff);
}

/* Force ALL text dark */
html, body, div, p, span, label {
    color: #111111 !important;
}

/* Headings */
h1, h2, h3, h4, h5, h6 {
    color: #111111 !important;
    font-weight: 700 !important;
}

/* Tabs */
button[data-baseweb="tab"] {
    color: #111111 !important;
    font-weight: 600 !important;
}

/* Sidebar text (if used later) */
section[data-testid="stSidebar"] * {
    color: #111111 !important;
}

/* Markdown containers */
[data-testid="stMarkdownContainer"] {
    color: #111111 !important;
}

/* Metric styling */
[data-testid="stMetricValue"] {
    color: #111111 !important;
    font-weight: bold;
}

[data-testid="stMetricLabel"] {
    color: #333333 !important;
}

/* Glass Card */
.glass-card {
    background: white;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    margin-bottom: 20px;
    color: #111111 !important;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = tf.keras.models.load_model("model.keras")

with open("class_indices.json") as f:
    class_indices = json.load(f)

class_names = list(class_indices.keys())

# ---------------- PRICE TABLE ----------------
prices = {
    "freshapples": 120,
    "freshbanana": 60,
    "freshoranges": 80,
    "rottenapples": 0,
    "rottenbanana": 0,
    "rottenoranges": 0
}

# ---------------- HEADER ----------------
st.markdown("""
<div class="glass-card">
<h1>🍎 FruitVision</h1>
<p>AI-Based Fresh vs Rotten Fruit Detection System</p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
col1.metric("Classes", len(class_names))
col2.metric("Accuracy", f"{MODEL_ACCURACY}%")
col3.metric("Model", "MobileNetV2")

tab1, tab2, tab3 = st.tabs(["🔍 Detect", "📊 Analytics", "📘 About"])

# ================= DETECT TAB =================
with tab1:

    input_method = st.radio(
        "Choose Input Method:",
        ["Upload Image", "Use Camera"],
        horizontal=True
    )

    image = None

    if input_method == "Upload Image":
        uploaded_file = st.file_uploader("Upload Fruit Image",
                                         type=["jpg", "png", "jpeg"])
        if uploaded_file:
            image = Image.open(uploaded_file)

    else:
        camera_image = st.camera_input("Take a Picture")
        if camera_image:
            image = Image.open(camera_image)

    if image:
        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Input Image", use_column_width=True)

        image_resized = image.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(image_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        probabilities = prediction[0] * 100
        predicted_class = class_names[np.argmax(prediction)]
        confidence = float(np.max(probabilities))
        price_value = prices.get(predicted_class, 0)

        with col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)

            if "fresh" in predicted_class:
                st.success("Fresh Fruit Detected")
            else:
                st.error("Rotten Fruit Detected")

            st.write(f"Prediction: {predicted_class}")
            st.progress(confidence / 100)
            st.write(f"Confidence: {round(confidence,2)}%")
            st.write(f"Estimated Market Price: ₹{price_value} per kg")

            st.markdown('</div>', unsafe_allow_html=True)

        # Probability Graph
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Prediction Confidence Distribution")

        fig_prob, ax_prob = plt.subplots()
        ax_prob.bar(class_names, probabilities)
        ax_prob.set_ylabel("Confidence (%)")
        ax_prob.set_xticks(range(len(class_names)))
        ax_prob.set_xticklabels(class_names, rotation=45, ha='right')
        ax_prob.set_ylim([0, 100])
        st.pyplot(fig_prob)

        st.markdown('</div>', unsafe_allow_html=True)

        if "history" not in st.session_state:
            st.session_state.history = []

        st.session_state.history.append({
            "Prediction": predicted_class,
            "Confidence": round(confidence,2),
            "Price": price_value
        })

# ================= ANALYTICS TAB =================
with tab2:

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Prediction History")

    if "history" in st.session_state and len(st.session_state.history) > 0:

        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df)

        st.subheader("Confidence Trend")

        fig_trend, ax_trend = plt.subplots()
        ax_trend.plot(
            range(1, len(df["Confidence"]) + 1),
            df["Confidence"],
            marker='o',
            linewidth=2
        )

        ax_trend.set_xlabel("Prediction Number")
        ax_trend.set_ylabel("Confidence (%)")
        ax_trend.set_ylim(0, 100)
        ax_trend.grid(True)

        st.pyplot(fig_trend)

    else:
        st.info("Upload an image to start analytics.")

    st.markdown('</div>', unsafe_allow_html=True)

# ================= ABOUT TAB =================
with tab3:

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    st.header("Project Overview")
    st.write("""
FruitVision is an end-to-end AI-powered fruit quality detection system.

It uses Transfer Learning (MobileNetV2) to classify fruits
as Fresh or Rotten using deep learning.
""")

    st.header("Features")
    st.write("""
- Real-time fruit classification  
- Camera support  
- Confidence visualization  
- Price estimation  
- Prediction history tracking  
- Interactive analytics dashboard  
""")

    st.header("Future Scope")
    st.write("""
- Supermarket automation  
- Mobile deployment  
- IoT-based fruit inspection  
- Smart inventory integration  
""")

    st.markdown('</div>', unsafe_allow_html=True)