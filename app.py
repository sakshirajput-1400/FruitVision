import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import json
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ---------------- CONFIG ----------------
IMG_SIZE = 224
MODEL_ACCURACY = 94.0

st.set_page_config(page_title="FruitVision", layout="wide")

# ---------------- CLEAN LIGHT UI ----------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: #f5f7fa;
}
.glass-card {
    background: white;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = tf.keras.models.load_model("model.keras")

with open("class_indices.json") as f:
    class_indices = json.load(f)

class_names = list(class_indices.keys())

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
<p>End-to-End AI Fruit Quality Detection Dashboard</p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
col1.metric("Classes", len(class_names))
col2.metric("Accuracy", f"{MODEL_ACCURACY}%")
col3.metric("Model", "MobileNetV2")

tab1, tab2, tab3 = st.tabs(["🔍 Detect", "📊 Analytics", "📘 About"])

# ================= DETECT =================
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
        predicted_class = class_names[np.argmax(prediction)]
        confidence = float(np.max(prediction))
        price_value = prices.get(predicted_class, 0)

        with col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)

            if "fresh" in predicted_class:
                st.success("Fresh Fruit Detected")
            else:
                st.error("Rotten Fruit Detected")

            st.write(f"Prediction: {predicted_class}")
            st.progress(confidence)
            st.write(f"Confidence: {round(confidence*100,2)}%")
            st.write(f"Estimated Price: ₹{price_value} per kg")

            st.markdown('</div>', unsafe_allow_html=True)

        if "history" not in st.session_state:
            st.session_state.history = []

        st.session_state.history.append({
            "Prediction": predicted_class,
            "Confidence": round(confidence*100,2),
            "Price": price_value
        })

# ================= ANALYTICS =================
with tab2:

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Prediction History")

    if "history" in st.session_state and len(st.session_state.history) > 0:

        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df)

        st.subheader("Confidence Trend")
        fig, ax = plt.subplots()
        ax.plot(df["Confidence"])
        ax.set_ylabel("Confidence %")
        ax.set_xlabel("Prediction Index")
        st.pyplot(fig)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Report",
            csv,
            "fruit_prediction_report.csv",
            "text/csv"
        )
    else:
        st.info("No predictions yet.")

    st.markdown('</div>', unsafe_allow_html=True)

    # -------- CONFUSION MATRIX BUTTON --------
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Model Evaluation")

    if st.button("Generate Confusion Matrix (Test Data)"):

        test_datagen = ImageDataGenerator(rescale=1./255)

        test_generator = test_datagen.flow_from_directory(
            'dataset/test',
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=32,
            class_mode='categorical',
            shuffle=False
        )

        y_true = test_generator.classes
        y_pred_probs = model.predict(test_generator)
        y_pred = np.argmax(y_pred_probs, axis=1)

        cm = confusion_matrix(y_true, y_pred)

        fig2, ax2 = plt.subplots(figsize=(8,6))
        sns.heatmap(cm,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    xticklabels=class_names,
                    yticklabels=class_names)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot(fig2)

    st.markdown('</div>', unsafe_allow_html=True)

# ================= ABOUT =================
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
    - Real-time prediction
    - Camera support
    - Confidence visualization
    - Price estimation
    - Prediction history tracking
    - Downloadable reports
    - Model evaluation dashboard
    """)

    st.header("Future Scope")
    st.write("""
    - Supermarket automation
    - Mobile deployment
    - IoT-based fruit inspection
    - Smart inventory integration
    """)

    st.markdown('</div>', unsafe_allow_html=True)