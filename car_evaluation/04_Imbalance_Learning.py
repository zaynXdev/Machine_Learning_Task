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
