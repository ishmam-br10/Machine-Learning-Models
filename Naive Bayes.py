#@title Naive bayes

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

X = independent
y = dependent
# Perform feature selection using SelectKBest and chi2
k_best = 9 # Select the top k features
selector = SelectKBest(chi2, k=k_best)
X_selected = selector.fit_transform(X, y)

# Split the selected data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.25, random_state=48) # highest 48, 12

# Initialize the GaussianNB classifier
gnb_classifier = GaussianNB()

# Define the hyperparameters to search for optimization (empty dictionary for Naive Bayes)
param_grid = {}

# Initialize GridSearchCV with the GaussianNB classifier and hyperparameter grid
grid_search = GridSearchCV(estimator=gnb_classifier, param_grid=param_grid, cv=10)

# Fit the model to the training data with hyperparameter tuning
grid_search.fit(X_train, y_train)

# Get the best model with the best hyperparameters
best_model = grid_search.best_estimator_

# Make predictions
y_pred = best_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Create confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Create classification report
class_report = classification_report(y_test, y_pred)

print("Best hyperparameters:", grid_search.best_params_)
print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)
# Generate and plot the confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='Blues', fmt='g', cbar=True)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix for Naive Bayes')
plt.show()