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
