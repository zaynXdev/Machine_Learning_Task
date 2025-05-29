import streamlit as st
import numpy as np
import joblib
from sklearn.datasets import load_breast_cancer

# # Load saved objects
# scaler = joblib.load('scaler_bc.pkl')
# agglo = joblib.load('agglo_bc.pkl')
# model = joblib.load('logreg_bc_model.pkl')
#
# # Get feature names
# data = load_breast_cancer()
# feature_names = data.feature_names
#
# st.title("Breast Cancer Prediction App")
# st.header("Input Features")
#
# # Collect input for all features
# input_values = []
# for feature in feature_names:
#     val = st.number_input(feature, value=float(data.data[:, list(feature_names).index(feature)].mean()))
#     input_values.append(val)
#
# input_data = np.array([input_values])
#
# # Preprocess input with scaler and agglomeration
# input_scaled = scaler.transform(input_data)
# input_agglo = agglo.transform(input_scaled)
#
# if st.button("Predict"):
#     pred = model.predict(input_agglo)[0]
#     st.write("## Prediction:", "Malignant" if pred == 0 else "Benign")
#



























import streamlit as st
import numpy as np
import joblib
from sklearn.datasets import load_breast_cancer
import pandas as pd

# Load saved objects
scaler = joblib.load('scaler_bc.pkl')
agglo = joblib.load('agglo_bc.pkl')
model = joblib.load('logreg_bc_model.pkl')

# Get feature names and data
data = load_breast_cancer()
feature_names = data.feature_names


# Find the most important feature from each cluster
def get_key_features(agglo, feature_names, n_clusters=5):
    cluster_labels = agglo.labels_
    key_features = []
    for i in range(n_clusters):
        # Get all features in this cluster
        cluster_features = np.array(feature_names)[cluster_labels == i]
        # For simplicity, just take the first feature in each cluster
        # In production, you might want to select based on importance scores
        key_features.append(cluster_features[0])
    return key_features


key_features = get_key_features(agglo, feature_names)
feature_indices = [list(feature_names).index(f) for f in key_features]

st.title("Breast Cancer Prediction App")
st.header("Input Key Features Only")

# Collect input for key features only
input_values = np.zeros(len(feature_names))  # Initialize with zeros
for i, feature in enumerate(key_features):
    default_val = float(data.data[:, feature_indices[i]].mean())
    val = st.number_input(feature, value=default_val,
                          min_value=float(data.data[:, feature_indices[i]].min()),
                          max_value=float(data.data[:, feature_indices[i]].max()))
    input_values[feature_indices[i]] = val

if st.button("Predict"):
    # Preprocess input with scaler and agglomeration
    input_scaled = scaler.transform([input_values])
    input_agglo = agglo.transform(input_scaled)

    pred = model.predict(input_agglo)[0]
    proba = model.predict_proba(input_agglo)[0]

    st.write("## Prediction:", "Malignant" if pred == 0 else "Benign")
    st.write(f"## Confidence: {proba[pred] * 100:.1f}%")

    # Show feature clusters info (optional)
    with st.expander("Feature Cluster Information"):
        st.write("We're using these key features from each cluster:")
        st.write(pd.DataFrame({
            "Cluster": range(5),
            "Key Feature": key_features
        }))


    #streamlit run breast_cancer_app.py