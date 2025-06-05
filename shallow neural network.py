import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

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
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#train the Shallow Neural Network model
model = MLPClassifier(hidden_layer_sizes=(20,), max_iter=1000, random_state=42)
model.fit(X_train, y_train)

#make predictions
y_pred = model.predict(X_test)

#evaluate the model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")

#create confusion matrix
cm = confusion_matrix(y_test, y_pred)

#display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['no', 'yes'])
disp.plot(cmap='Blues', values_format='d', xticks_rotation='horizontal')

#add title to the plot
plt.title("Confusion Matrix - Shallow Neural Network Model", fontsize=12, fontweight="bold")
#show the plot
plt.show()
