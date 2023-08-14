#@title Support Vector Machine (SVM)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Split your data into features (X) and target (y)
X = independent
y = dependent

# Split your data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=39)

# Initialize the SVM classifier
svm_clf = SVC()

# Define hyperparameters and their values for tuning
param_grid = {
    'C': [0.1, 1, 10],  # Regularization parameter
    'kernel': ['linear', 'rbf']  # Kernel type: linear or radial basis function (RBF)
}

# Initialize GridSearchCV for hyperparameter tuning and cross-validation
grid_search = GridSearchCV(estimator=svm_clf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Train the model on the training data and find the best hyperparameters
grid_search.fit(X_train, y_train)

# Get the best estimator after hyperparameter tuning
best_svm_clf = grid_search.best_estimator_

# Make predictions on the test set using the best model
y_pred = best_svm_clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of the SVM model:", accuracy)

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
plt.title("Confusion Matrix for SVM")
plt.show()