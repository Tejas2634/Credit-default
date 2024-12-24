# Credit-default
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


data = {
    'Income': np.random.randint(20000, 100000, 4000),
    'Age': np.random.randint(18, 70, 4000),
    'Loan': np.random.randint(5000, 50000, 4000),
}
data['Loan_to_Income'] = data['Loan'] / data['Income']
data['Default'] = np.random.choice([0, 1], size=4000, p=[0.7, 0.3])

df = pd.DataFrame(data)


X = df[['Income', 'Age', 'Loan', 'Loan_to_Income']]
y = df['Default']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)


y_pred = model.predict(X_test_scaled)


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


feature_importances = pd.Series(model.feature_importances_, index=X.columns)
print("\nFeature Importances:")
print(feature_importances.sort_values(ascending=False))
