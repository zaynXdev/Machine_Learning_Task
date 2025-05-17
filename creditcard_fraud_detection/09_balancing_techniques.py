import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# 1. Load Dataset
df = pd.read_csv('creditcard.csv')

# 2. Check class distribution (highly imbalanced)
print("                                                                     ")
print("---------------------------------------------------------------------------------")
print("                  Class Distribution ")
print("---------------------------------------------------------------------------------")
print( df['Class'].value_counts())
print("                                                                     ")
print("Percentage of Fraudulent Transactions: {:.4f}%".format(
    100 * df['Class'].sum() / len(df)
))

# --- FAST SUBSET FOR DEVELOPMENT ---
df_small = pd.concat([
    df[df['Class'] == 0].sample(n=3000, random_state=42),  # 3,000 normal
    df[df['Class'] == 1]
])
df = df_small.sample(frac=1, random_state=42)

# 3. Split features and target
X = df.drop(['Class', 'Time'], axis=1)  # 'Time' is usually not predictive
y = df['Class']

# 4. Split into train & test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 5. Train and Evaluate BEFORE Balancing (Random Forest)
print("                                                                     ")
print("---------------------------------------------------------------------------------")
print("                  BEFORE Balancing ")
print("---------------------------------------------------------------------------------")
rf = RandomForestClassifier(random_state=42, n_estimators=100)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("                                                                     ")
print("---------------------------------------------------------------------------------")
print("                  Random Forest ")
print("---------------------------------------------------------------------------------")
print( classification_report(y_test, y_pred_rf, digits=4))

# 6. Train and Evaluate BEFORE Balancing (Logistic Regression)
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print("                                                                     ")
print("---------------------------------------------------------------------------------")
print("                  Logistic Regression ")
print("---------------------------------------------------------------------------------")
print( classification_report(y_test, y_pred_lr, digits=4))

# 7. Apply SMOTE (Over-sampling)
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print("                                                                     ")
print("---------------------------------------------------------------------------------")
print("                After SMOTE new class distribution ")
print("---------------------------------------------------------------------------------")
print( np.bincount(y_train_sm))

# 8. Retrain after SMOTE (Random Forest)
rf.fit(X_train_sm, y_train_sm)
y_pred_rf_sm = rf.predict(X_test)

print("                                                                     ")
print("---------------------------------------------------------------------------------")
print("                  AFTER SMOTE (Random Forest) ")
print("---------------------------------------------------------------------------------")
print(classification_report(y_test, y_pred_rf_sm, digits=4))

# 9. Apply Random Under-Sampling
rus = RandomUnderSampler(random_state=42)
X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)
print("                                                                     ")
print("---------------------------------------------------------------------------------")
print("             After Random Under Sampling  new class distribution ")
print("---------------------------------------------------------------------------------")
print( np.bincount(y_train_rus))

# 10. Retrain after Under-Sampling (Random Forest)
rf.fit(X_train_rus, y_train_rus)
y_pred_rf_rus = rf.predict(X_test)

print("                                                                     ")
print("---------------------------------------------------------------------------------")
print("                  AFTER RANDOM UNDER-SAMPLING (Random Forest) ")
print("---------------------------------------------------------------------------------")
print(classification_report(y_test, y_pred_rf_rus, digits=4))

# Optional: Compare with Logistic Regression after balancing
lr.fit(X_train_sm, y_train_sm)
y_pred_lr_sm = lr.predict(X_test)

print("                                                                     ")
print("---------------------------------------------------------------------------------")
print("                AFTER SMOTE (Logistic Regression) ")
print("---------------------------------------------------------------------------------")
print(classification_report(y_test, y_pred_lr_sm, digits=4))

lr.fit(X_train_rus, y_train_rus)
y_pred_lr_rus = lr.predict(X_test)

print("                                                                     ")
print("---------------------------------------------------------------------------------")
print("               AFTER RANDOM UNDER-SAMPLING (Logistic Regression) ")
print("---------------------------------------------------------------------------------")
print(classification_report(y_test, y_pred_lr_rus, digits=4))

# 11. (Optional) Confusion Matrices for additional insight
print("\nConfusion Matrix BEFORE Balancing (Random Forest):")
print(confusion_matrix(y_test, y_pred_rf))
print("\nConfusion Matrix AFTER SMOTE (Random Forest):")
print(confusion_matrix(y_test, y_pred_rf_sm))
print("\nConfusion Matrix AFTER Random Under-Sampling (Random Forest):")
print(confusion_matrix(y_test, y_pred_rf_rus))