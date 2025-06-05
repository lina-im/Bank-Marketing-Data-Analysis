import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

#load the dataset
df = pd.read_csv("C:/Users/User/OneDrive/Documents/Desktop/bank-full.csv", sep=';') 

#convert binary categorical variables ('yes' -> 1, 'no' -> 0)
binary_columns = ['default', 'housing', 'loan', 'y']
for col in binary_columns:
    df[col] = df[col].map({'yes': 1, 'no': 0})

#one-hot encode categorical variables
df = pd.get_dummies(df, columns=['job', 'marital', 'education', 'contact', 'month', 'poutcome'], drop_first=True)

#define input (X) and output (y)
X = df.drop(columns=['y'])  # Features
y = df['y']  # Target variable

#standardize numeric features for better model performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.4, random_state=42)

#train the Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

#make predictions
y_pred = model.predict(X_test)

#evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")

#compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

#plot the confusion matrix using seaborn heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
plt.xlabel('Predicted')
plt.ylabel('Observed')
plt.title('Confusion Matrix')
plt.show()
