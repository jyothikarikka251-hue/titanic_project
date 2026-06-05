import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -------------------------
# 1. Load Dataset
# -------------------------
df = pd.read_csv("train_data.csv")

print(df.head())

# -------------------------
# 2. Data Cleaning
# -------------------------

# Fill missing Age with median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill Embarked with mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop Cabin (too many missing values)
df.drop(columns=['Cabin'], inplace=True)

# Drop rows with missing Fare (if any)
df['Fare'].fillna(df['Fare'].median(), inplace=True)

# -------------------------
# 3. Feature Engineering
# -------------------------

# Convert Sex to numeric
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Convert Embarked to numeric
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Select features
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

X = df[features]
y = df['Survived']

# -------------------------
# 4. Train-Test Split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# 5. Model Training
# -------------------------
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# -------------------------
# 6. Predictions
# -------------------------
y_pred = model.predict(X_test)

# -------------------------
# 7. Evaluation
# -------------------------
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -------------------------
# 8. Simple Visualization
# -------------------------

plt.figure()
plt.scatter(X_test['Age'], y_test, label="Actual", alpha=0.6)
plt.scatter(X_test['Age'], y_pred, label="Predicted", alpha=0.6)
plt.xlabel("Age")
plt.ylabel("Survived (0/1)")
plt.title("Titanic Survival Prediction")
plt.legend()
plt.show()