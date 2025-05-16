import pandas as pd

# Load the dataset
df = pd.read_csv('car_evaluation.csv')




print("                                                                     ")
print("                       .1  -- PreProcessing --")

# 1. Display columns and datatypes
print("                                                                     ")
print("---------------------------------------------------------------------------------")
print("                         Column Names and Data Types")
print("---------------------------------------------------------------------------------")
print("Columns in the dataset:", df.columns.tolist())
print("\nData types:\n", df.dtypes)



# 2. Show unique values for each column (to check categorical levels)
print("                                                                     ")
print("---------------------------------------------------------------------------------")
print("                        unique values for each column")
print("---------------------------------------------------------------------------------")
for col in df.columns:
    print(f"\nUnique values in '{col}': {df[col].unique()}")



# 3. Check for missing values
print("                                                                     ")
print("---------------------------------------------------------------------------------")
print("                                 missing values")
print("---------------------------------------------------------------------------------")
print("\nMissing values per column:\n", df.isnull().sum())



# Check which columns are categorical (object dtype or few unique values)
print("                                                                     ")
print("---------------------------------------------------------------------------------")
print("                                 Categorical Columns")
print("---------------------------------------------------------------------------------")
categorical_columns = []
for col in df.columns:
    n_unique = df[col].nunique()
    dtype = df[col].dtype
    print(f"Column: {col} | Unique values: {n_unique} | Dtype: {dtype}")
    # If dtype is object or number of unique values is small (e.g., <=10), consider categorical
    if dtype == 'object' or n_unique <= 10:
        categorical_columns.append(col)
print("\nCategorical columns:")
print(categorical_columns)



# 4. Encode all categorical columns using Label Encoding (since all are categorical)
print("                                                                     ")
print("---------------------------------------------------------------------------------")
print("                           Encoded categorical columns")
print("---------------------------------------------------------------------------------")
from sklearn.preprocessing import LabelEncoder

df_encoded = df.copy()
label_encoders = {}
for col in df.columns:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    print(f"Label encoding for '{col}': {dict(zip(le.classes_, le.transform(le.classes_)))}")



# 5. Save encoded dataframe for further analysis
print("                                                                     ")
print("---------------------------------------------------------------------------------")
print("                           Saved encoded dataframe")
print("---------------------------------------------------------------------------------")
df_encoded.to_csv('car_evaluation_encoded.csv', index=False)
print("\nPreprocessing done. Encoded dataset saved as 'car_evaluation_encoded.csv'.")




print("                                                                     ")
print("                                                                     ")
print("                 .2  -- Unsupervised learning techniques  --")
print("                                                                     ")

import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score

# Load the encoded dataset
df = pd.read_csv('car_evaluation_encoded.csv')

# Separate features and true labels (which we will NOT use for clustering but will use for ARI validation)
X = df.drop('class', axis=1)
y_true = df['class']

print("Applying KMeans clustering...")
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans_labels = kmeans.fit_predict(X)

print("Applying Agglomerative Clustering...")
agglo = AgglomerativeClustering(n_clusters=4)
agglo_labels = agglo.fit_predict(X)

# Cluster validation metrics
print("                                                                     ")
print("---------------------------------------------------------------------------------")
print("                           KMeans Validation ")
print("---------------------------------------------------------------------------------")
print("Silhouette Score:", silhouette_score(X, kmeans_labels))
print("Davies-Bouldin Score:", davies_bouldin_score(X, kmeans_labels))
print("Adjusted Rand Index vs. true labels:", adjusted_rand_score(y_true, kmeans_labels))


print("                                                                     ")
print("---------------------------------------------------------------------------------")
print("                          Agglomerative Clustering Validation ")
print("---------------------------------------------------------------------------------")
print("Silhouette Score:", silhouette_score(X, agglo_labels))
print("Davies-Bouldin Score:", davies_bouldin_score(X, agglo_labels))
print("Adjusted Rand Index vs. true labels:", adjusted_rand_score(y_true, agglo_labels))


print("                                                                     ")
print("                                                                     ")
print("                 .3  -- Feature Selection techniques  --")
print("                                                                     ")

import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2

# Load the encoded dataset
df = pd.read_csv('car_evaluation_encoded.csv')

# Separate features and target
X = df.drop('class', axis=1)
y = df['class']

# Apply Chi-Square Feature Selection (suitable for categorical data)
selector = SelectKBest(score_func=chi2, k='all')
selector.fit(X, y)

# Show scores for each feature
scores = selector.scores_
features = X.columns
print("                                                                     ")
print("---------------------------------------------------------------------------------")
print("                          Chi Square Scores for each feature ")
print("---------------------------------------------------------------------------------")
for feat, score in zip(features, scores):
    print(f"{feat}: {score:.2f}")

# Select top 4 features (as an example; you can select any number based on scores)
k = 4
top_indices = selector.get_support(indices=True)[:k]
selected_features = [features[i] for i in top_indices]
print("                                                                     ")
print("---------------------------------------------------------------------------------")
print("                    Selected features to use for further experiments ")
print("---------------------------------------------------------------------------------")
print( selected_features)

# Save the reduced dataframe
df_selected = df[selected_features + ['class']]
df_selected.to_csv('car_evaluation_selected_( Reduced ).csv', index=False)
print("                                                                     ")
print("---------------------------------------------------------------------------------")
print("                    Saved Reduced Dataset ")
print("---------------------------------------------------------------------------------")
print("\n 'car_evaluation_selected_( Reduced ).csv'.")


print("                                                                     ")
print("                                                                     ")
print("      ||   .4  -- Imbalance learning techniques  --   ||")
print("                                                                     ")

import pandas as pd
from collections import Counter
from imblearn.over_sampling import SMOTE, RandomOverSampler

# Load the selected features dataset
df = pd.read_csv('car_evaluation_selected_( Reduced ).csv')
X = df.drop('class', axis=1)
y = df['class']

# Check the original class distribution
print("                                                                     ")
print("---------------------------------------------------------------------------------")
print("                    Original class distribution ")
print("---------------------------------------------------------------------------------")
print("Original class distribution:", Counter(y))

# --- Apply SMOTE (Synthetic Minority Over-sampling Technique) ---
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X, y)
print("                                                                     ")
print("---------------------------------------------------------------------------------")
print("                    Class distribution after SMOTE ")
print("---------------------------------------------------------------------------------")
print( Counter(y_smote))

# --- Apply Random OverSampler ---
ros = RandomOverSampler(random_state=42)
X_ros, y_ros = ros.fit_resample(X, y)
print("                                                                     ")
print("---------------------------------------------------------------------------------")
print("                  Class distribution after RandomOverSampler ")
print("---------------------------------------------------------------------------------")
print( Counter(y_ros))

# Save the SMOTE-balanced dataset for further steps
df_smote = pd.DataFrame(X_smote, columns=X.columns)
df_smote['class'] = y_smote
df_smote.to_csv('car_evaluation_balanced.csv', index=False)
print("                                                                     ")
print("---------------------------------------------------------------------------------")
print("                  Saved SMOTE balanced dataset ")
print("---------------------------------------------------------------------------------")
print("\n 'car_evaluation_balanced.csv'.")


print("                                                                     ")
print("                                                                     ")
print("      ||   .5  -- Supervised learning classifiers Application  --   ||")
print("                                                                     ")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier

# Load the balanced dataset (from SMOTE)
df = pd.read_csv('car_evaluation_balanced.csv')
X = df.drop('class', axis=1)
y = df['class']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 1. Logistic Regression (linear)
clf1 = LogisticRegression(max_iter=1000, random_state=42)
clf1.fit(X_train, y_train)
pred1 = clf1.predict(X_test)
print("                                                                     ")
print("---------------------------------------------------------------------------------")
print("                  Logistic Regression ")
print("---------------------------------------------------------------------------------")
print(classification_report(y_test, pred1))

# 2. Decision Tree (non-linear)
clf2 = DecisionTreeClassifier(random_state=42)
clf2.fit(X_train, y_train)
pred2 = clf2.predict(X_test)
print("                                                                     ")
print("---------------------------------------------------------------------------------")
print("                  Decision Tree ")
print("---------------------------------------------------------------------------------")
print(classification_report(y_test, pred2))

# 3. Random Forest (non-linear, ensemble)
clf3 = RandomForestClassifier(random_state=42)
clf3.fit(X_train, y_train)
pred3 = clf3.predict(X_test)

print("                                                                     ")
print("---------------------------------------------------------------------------------")
print("                  Random Forest")
print("---------------------------------------------------------------------------------")
print(classification_report(y_test, pred3))

# 4. Gradient Boosting (boosting)
clf4 = GradientBoostingClassifier(random_state=42)
clf4.fit(X_train, y_train)
pred4 = clf4.predict(X_test)
print("                                                                     ")
print("---------------------------------------------------------------------------------")
print("                  Gradient Boosting ")
print("---------------------------------------------------------------------------------")
print(classification_report(y_test, pred4))

# 5. AdaBoost (boosting)
clf5 = AdaBoostClassifier(random_state=42)
clf5.fit(X_train, y_train)
pred5 = clf5.predict(X_test)
print("                                                                     ")
print("---------------------------------------------------------------------------------")
print("                  AdaBoost ")
print("---------------------------------------------------------------------------------")
print(classification_report(y_test, pred5))

# Optionally, print a summary of accuracy scores for comparison
print("                                                                     ")
print("---------------------------------------------------------------------------------")
print("                  Accuracy Scores ")
print("---------------------------------------------------------------------------------")
print("Logistic Regression  :  ", accuracy_score(y_test, pred1))
print("Decision Tree        :  ", accuracy_score(y_test, pred2))
print("Random Forest        :  ", accuracy_score(y_test, pred3))
print("Gradient Boosting    :  ", accuracy_score(y_test, pred4))
print("AdaBoost             :  ", accuracy_score(y_test, pred5))