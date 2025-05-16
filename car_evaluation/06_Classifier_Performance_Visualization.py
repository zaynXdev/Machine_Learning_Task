import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Load the balanced dataset (from SMOTE)
df = pd.read_csv('car_evaluation_balanced.csv')
X = df.drop('class', axis=1)
y = df['class']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42)
}

results = []
reports = {}
conf_matrices = {}

for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    results.append({'Classifier': name, 'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-Score': f1})
    reports[name] = classification_report(y_test, y_pred)
    conf_matrices[name] = confusion_matrix(y_test, y_pred)

# Create a DataFrame to summarize results
results_df = pd.DataFrame(results)
print("                                                                     ")
print("---------------------------------------------------------------------------------")
print("                  Performance Comparison Table ")
print("---------------------------------------------------------------------------------")
print(results_df)

# Plot bar chart for performance metrics
plt.figure(figsize=(12, 7))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
for idx, metric in enumerate(metrics):
    plt.subplot(2, 2, idx + 1)
    sns.barplot(x='Classifier', y=metric, data=results_df)
    plt.title(f'{metric} by Classifier')
    plt.ylim(0, 1)
    plt.xticks(rotation=20)

plt.tight_layout()
plt.show()

# Print classification report for each classifier
for name in classifiers:
    print("                                                                     ")
    print("---------------------------------------------------------------------------------")
    print(f"\t\tclassification report of {name}  ")
    print("---------------------------------------------------------------------------------")
    print(reports[name])

# Plot confusion matrix for each classifier
for name in classifiers:
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrices[name], annot=True, fmt='d', cmap='Blues',
                xticklabels=sorted(y.unique()), yticklabels=sorted(y.unique()))
    plt.title(f'Confusion Matrix: {name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()