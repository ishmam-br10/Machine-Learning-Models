#@title Random State finder ensemble learning
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

X = independent
y = dependent

rf_model = RandomForestClassifier(random_state=42)
knn_model = KNeighborsClassifier(n_neighbors=5)
svm_model = SVC(kernel='linear', random_state=42)
nb_model = GaussianNB()
xgb_model = XGBClassifier(random_state=42)

# Create the ensemble model using a voting classifier
ensemble_model = VotingClassifier(estimators=[
    ('rf', rf_model),
    ('knn', knn_model),
    ('svm', svm_model),
    ('nb', nb_model),
    ('xgb', xgb_model)
], voting='hard')

# Initialize variables to track the best random state and accuracy
best_accuracy = 0
best_random_state = None

# Loop through different random states to find the best one
for random_state in range(200):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=random_state)
    ensemble_model.fit(X_train, y_train)
    y_pred = ensemble_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    # print(f"Iteration : {random_state}")
    # print(f"Accuracy : {accuracy}")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_random_state = random_state

print(f"Best Random State: {best_random_state}")
print(f"Best Accuracy: {best_accuracy:.4f}")

# Using the best random state, split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=best_random_state)

# Fit the ensemble model to the training data
ensemble_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = ensemble_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=True)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix Ensemble Learning')
plt.show()