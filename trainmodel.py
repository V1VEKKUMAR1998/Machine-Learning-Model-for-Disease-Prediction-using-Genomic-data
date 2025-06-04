import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load preprocessed data
X_train = pd.read_csv('X_train_preprocessed.csv')
X_test = pd.read_csv('X_test_preprocessed.csv')
y_train = pd.read_csv('y_train.csv')
y_test = pd.read_csv('y_test.csv')

# Initialize and train model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train.values.ravel())

# Predict and evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save model predictions
pd.DataFrame({'SampleID': range(len(y_test)), 'Predicted_disease_risk': y_pred}).to_csv('predictions.csv', index=False)
print("Predictions saved to predictions.csv")