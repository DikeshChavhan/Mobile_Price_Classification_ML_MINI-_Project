import streamlit as st
import pandas as pd
import pickle

st.title("🧠 ML Model Prediction App")

st.write("""
### Upload your dataset and get instant predictions using a pre-trained model (.pkl)
""")

# Load pre-trained model
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    st.success("✅ Model loaded successfully!")
except:
    st.error("⚠️ Model file not found! Please ensure model.pkl is in the same folder.")

# Upload CSV for prediction
uploaded_file = st.file_uploader("📂 Upload your CSV file for prediction", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("📊 Uploaded Data")
    st.dataframe(data.head())

    # Predict
    if st.button("🚀 Predict"):
        predictions = model.predict(data)
        st.subheader("🔮 Predictions")
        st.write(predictions)
        st.download_button(
            label="💾 Download Predictions",
            data=pd.DataFrame(predictions, columns=["Predictions"]).to_csv(index=False),
            file_name="predictions.csv",
            mime="text/csv"
        )
