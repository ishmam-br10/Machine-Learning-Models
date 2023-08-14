#@title XG Boost

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

X= independent
y = dependent

# Split the data into training and test sets
# [(0.6648936170212766, 0), (0.7021276595744681, 4), (0.7287234042553191, 39),
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=39)

# Create an XGBoost classifier
xgb_classifier = xgb.XGBClassifier(
    objective='binary:logistic',  # for binary classification
    n_estimators=100,  # number of trees (boosting rounds) #100
    learning_rate=0.1,  # step size shrinkage used in update to prevent overfitting #0.1
    max_depth=3,  # maximum depth of a tree # 3
    random_state=42
)

# Train the XGBoost model on the training data
xgb_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = xgb_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix xg Boost')
plt.show()