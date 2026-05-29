import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle

# 1. தரவை ஏற்றுதல்
df = pd.read_csv('diabetes.csv')

X = df.drop(columns='Outcome')
y = df['Outcome']

# 2. Train Test Split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# 3. Missing Value Imputation (Data Leakage இல்லாமல்)
columns_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

for col in columns_with_zeros:
    # Train Data
    x_train[col] = x_train[col].replace(0, np.nan)
    median_value = x_train[col].median()
    x_train[col] = x_train[col].fillna(median_value)

    # Test Data (Train median-ஐப் பயன்படுத்த வேண்டும்)
    x_test[col] = x_test[col].replace(0, np.nan)
    x_test[col] = x_test[col].fillna(median_value)

# 4. Feature Scaling
scaled = StandardScaler()
x_train_scaled = scaled.fit_transform(x_train)
x_test_scaled = scaled.transform(x_test)

# 5. Hyperparameter Tuning (GridSearchCV)
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 4, 5, 6, 7, 8],
    'min_samples_split': [3, 4, 5, 6, 7, 8],
    'min_samples_leaf': [1, 2, 4]
}

dt_model = DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(estimator=dt_model, param_grid=param_grid, cv=5, scoring='recall')
grid_search.fit(x_train_scaled, y_train)

best_dt_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)
print("-" * 40)

# 6. Prediction & Evaluation
y_pred_tuned = best_dt_model.predict(x_test_scaled)

acc = accuracy_score(y_test, y_pred_tuned)
prec = precision_score(y_test, y_pred_tuned)
rec = recall_score(y_test, y_pred_tuned)
f1 = f1_score(y_test, y_pred_tuned)

print("=== Performance Metrics ===")
print(f"Accuracy:  {acc:.2%}")
print(f"Precision: {prec:.2%}")
print(f"Recall:    {rec:.2%}")
print(f"F1-Score:  {f1:.2%}\n")

print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred_tuned))

# 7. Model & Scaler சேமித்தல்
model_package = {
    "scaler": scaled,
    "model": best_dt_model
}

with open('diabetes_model_package.pkl', "wb") as file_obj:
    pickle.dump(model_package, file_obj)
print("\nModel and Scaler successfully saved!")


print(x_train[col])