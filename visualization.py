import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# Load preprocessed data
X_train = pd.read_csv('X_train_preprocessed.csv')
y_train = pd.read_csv('y_train.csv').values.ravel()

# Train Random Forest model
rf = RandomForestClassifier(random_state=42, n_estimators=100)
rf.fit(X_train, y_train)

# Get feature importance
feature_importance = pd.DataFrame({
    'SNP': X_train.columns,
    'Importance': rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.bar(feature_importance['SNP'][:10], feature_importance['Importance'][:10])
plt.xticks(rotation=45)
plt.xlabel('SNP')
plt.ylabel('Feature Importance')
plt.title('Top 10 SNP Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()
print("Feature importance plot saved as feature_importance.png")