#@title K Nearest Neighbour

import pandas as pd
import numpy as np  # Import NumPy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



# Split your data into features (X) and target (y)
X = independent
y = dependent

# Convert the DataFrame to a NumPy array
X = X.values
y = y.values

# Split your data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)

# Initialize the KNN classifier
knn_clf = KNeighborsClassifier()

# Define hyperparameters and their values for tuning
param_grid = {
  'n_neighbors': [3, 5, 7],  # Number of neighbors to consider
  'weights': ['uniform', 'distance'],  # Weight function used in prediction
  'p': [1, 2],  # Power parameter for the Minkowski metric (1 for Manhattan, 2 for Euclidean)
}

# Initialize GridSearchCV for hyperparameter tuning and cross-validation
grid_search = GridSearchCV(estimator=knn_clf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Train the model on the training data and find the best hyperparameters
grid_search.fit(X_train, y_train)

# Get the best estimator after hyperparameter tuning
best_knn_clf = grid_search.best_estimator_

# Make predictions on the test set using the best model
y_pred = best_knn_clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of the KNN model:", accuracy)

# Print the confusion matrix and classification report
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot the confusion matrix
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='Blues', fmt='g', cbar=True)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix for KNN")
plt.show()