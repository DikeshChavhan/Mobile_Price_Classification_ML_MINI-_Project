import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.title("🧠 Machine Learning Classification App")

st.write("""
### This Streamlit app allows you to upload a dataset and train ML models (Logistic Regression, KNN, and SVM)
""")

# File uploader
uploaded_file = st.file_uploader("📂 Upload your CSV dataset", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("📊 Dataset Preview")
    st.dataframe(data.head())

    # Select target column
    target = st.selectbox("🎯 Select Target Column", data.columns)

    # Feature selection
    X = data.drop(columns=[target])
    y = data[target]

    # Train-test split
    test_size = st.slider("Select Test Size", 0.1, 0.5, 0.2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    st.write("✅ Data split into training and testing sets")

    # Model selection
    model_name = st.selectbox("Select Model", ["Logistic Regression", "KNN", "SVM"])

    if st.button("🚀 Train Model"):
        if model_name == "Logistic Regression":
            model = LogisticRegression()
        elif model_name == "KNN":
            model = KNeighborsClassifier()
        else:
            model = SVC()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        st.success(f"Model Trained Successfully! ✅")
        st.write(f"**Accuracy:** {acc:.2f}")

        # Display confusion matrix
        st.subheader("📉 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        st.write(cm)

        st.subheader("📋 Classification Report")
        st.text(classification_report(y_test, y_pred))
else:
    st.info("👆 Please upload a dataset to begin.")
