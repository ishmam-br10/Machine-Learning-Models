#@title Random Forest
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


X = independent
y =dependent
#[(0, 0.6861702127659575), (14, 0.6914893617021277), (15, 0.7127659574468085), (38, 0.7180851063829787), (91, 0.723404255319149), (101, 0.7287234042553191)]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=101)

# Initialize the Random Forest classifier
rf_clf = RandomForestClassifier(random_state=42)

# Define hyperparameters and their values for tuning
param_grid = {
    'n_estimators': [50, 100, 200],  # Number of trees in the forest
    'max_depth': [None, 10, 20],     # Maximum depth of the tree
    'min_samples_split': [2, 5, 10], # Minimum number of samples required to split an internal node
}

# Initialize GridSearchCV for hyperparameter tuning and cross-validation
grid_search = GridSearchCV(estimator=rf_clf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Train the model on the training data and find the best hyperparameters
grid_search.fit(X_train, y_train)

# Get the best estimator after hyperparameter tuning
best_rf_clf = grid_search.best_estimator_

# Make predictions on the test set using the best model
y_pred = best_rf_clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of the Random Forest model:", accuracy)

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
plt.title("Confusion Matrix for Random Forest")
plt.show()