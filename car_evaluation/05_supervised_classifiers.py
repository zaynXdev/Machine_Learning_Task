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