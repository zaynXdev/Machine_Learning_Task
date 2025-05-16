import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer  # Example DS2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import FeatureAgglomeration
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Standardize features for feature extraction
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2A. Feature Extraction: PCA
pca = PCA(n_components=5, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# 2B. Feature Extraction: Feature Agglomeration
agglo = FeatureAgglomeration(n_clusters=5)
X_agglo = agglo.fit_transform(X_scaled)

# 3. Classification (using Logistic Regression as example)
def classify_and_report(X_features, y, method_name):
    X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42, stratify=y)
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("                                                                     ")
    print("---------------------------------------------------------------------------------")
    print(f"\t\t {method_name} ")
    print("---------------------------------------------------------------------------------")

    print("Accuracy:", acc)
    print(classification_report(y_test, y_pred))
    return acc



acc_pca = classify_and_report(X_pca, y, "PCA Feature Extraction")
acc_agglo = classify_and_report(X_agglo, y, "Feature Agglomeration Feature Extraction")

print(f"\nSummary:\nPCA Accuracy: {acc_pca}\nFeature Agglomeration Accuracy: {acc_agglo}")