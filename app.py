import streamlit as st
import pandas as pd
import pickle

# --- App Title ---
st.title("🏠 Price Prediction Web App")
st.write("""
This app uses a pre-trained **Linear Regression model** to predict prices based on your input dataset.
Upload your CSV file (with same features as the training data, excluding the 'price' column) and get predictions instantly!
""")

# --- Load Pre-trained Model ---
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    st.success("✅ Model loaded successfully!")
except FileNotFoundError:
    st.error("⚠️ Model file not found! Please make sure 'model.pkl' is in the same folder as this script.")
    st.stop()

# --- File Upload Section ---
uploaded_file = st.file_uploader("📂 Upload your dataset for prediction (CSV format)", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded CSV
    data = pd.read_csv(uploaded_file)

    st.subheader("📊 Uploaded Data Preview")
    st.dataframe(data.head())

    # --- Predict Prices ---
    if st.button("🚀 Predict Prices"):
        try:
            predictions = model.predict(data)

            # Display predictions
            st.subheader("💰 Predicted Prices")
            st.write(predictions)

            # Add predictions to the dataframe
            data["Predicted_Price"] = predictions

            # Allow user to download results
            csv = data.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="💾 Download Predictions as CSV",
                data=csv,
                file_name="predicted_prices.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"⚠️ Prediction failed: {e}")
else:
    st.info("👆 Upload a CSV file to start predicting.")
