# Step 1: Data Processing 
import pandas as pandas

df=pandas.read_csv('Project_1_Data.csv')

#Step 2: Data Visualization

print("")
print("Statistical Analysis")
print(df[['X', 'Y', 'Z']].describe())

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.hist(df['X'], bins=20, color='red',edgecolor='white')
plt.title('X-Coordinate Distribution')
plt.xlabel("X-Value")
plt.ylabel("Frequency")

# Plot for Y coordinate
plt.subplot(1, 3, 2)
plt.hist(df['Y'], bins=20, color='green',edgecolor='white')
plt.title('Y-Coordinate Distribution')
plt.xlabel("Y-Value")
plt.ylabel("Frequency")

# Plot for Z coordinate
plt.subplot(1, 3, 3)
plt.hist(df['Z'], bins=20, color='blue',edgecolor='white')
plt.title('Z-Coordinate Distribution')
plt.xlabel("Z-Value")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

#Step 3: Correlation Matrix

import seaborn as sns

correlation_matrix = df.corr(method='pearson')

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, linewidths=0.5)

plt.title('Correlation Matrix')

plt.show()

#Step 4: Classification Model Development/Engineering

X = df[['X', 'Y', 'Z']]
y = df['Step']

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

random_forest = RandomForestClassifier()
log_reg = LogisticRegression(max_iter=20000)
svc = SVC()

param_grid_rf = {'n_estimators': [10, 50, 100], 'max_depth': [5, 10, 20]}
param_grid_log_reg = {'C': [0.1, 1, 10]}
param_grid_svc = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}

grid_search_rf = GridSearchCV(random_forest, param_grid_rf, cv=5)
grid_search_rf.fit(X_train, y_train)

grid_search_log_reg = GridSearchCV(log_reg, param_grid_log_reg, cv=5)
grid_search_log_reg.fit(X_train, y_train)

grid_search_svc = GridSearchCV(svc, param_grid_svc, cv=5)
grid_search_svc.fit(X_train, y_train)

random_search_rf = RandomizedSearchCV(random_forest, param_distributions=param_grid_rf, n_iter=5, cv=5, random_state=42)
random_search_rf.fit(X_train, y_train)

y_pred_log_reg = grid_search_log_reg.predict(X_test)
y_pred_rf = grid_search_rf.predict(X_test)
y_pred_svc = grid_search_svc.predict(X_test)

from sklearn.metrics import accuracy_score

acc_log_reg = accuracy_score(y_test, y_pred_log_reg)
acc_rf = accuracy_score(y_test, y_pred_rf)
acc_svc = accuracy_score(y_test, y_pred_svc)

print("")
print(f'Logistic Regression Accuracy: {acc_log_reg}')
print(f'Random Forest Accuracy: {acc_rf}')
print(f'SVC Accuracy: {acc_svc}')
print("")

#Step 5: Model Performance Analysis

from sklearn.metrics import classification_report, confusion_matrix

print("Logistic Regression Classification Report")
print(classification_report(y_test, y_pred_log_reg))

print("Random Forest Classification Report")
print(classification_report(y_test, y_pred_rf))

print("SVC Classification Report")
print(classification_report(y_test, y_pred_svc))

conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='Greens', cbar=False)
plt.title('Confusion Matrix for Random Forest')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

#Step 6: Stacked Model Performence Analysis

from sklearn.ensemble import StackingClassifier

estimators = [
    ('rf', grid_search_rf.best_estimator_),  
    ('svc', grid_search_svc.best_estimator_)
]

stacked_model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())

stacked_model.fit(X_train, y_train)

y_pred_stacked = stacked_model.predict(X_test)

print("Stacked Model Classification Report")
print(classification_report(y_test, y_pred_stacked))

conf_matrix_stacked = confusion_matrix(y_test, y_pred_stacked)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix_stacked, annot=True, fmt='d', cmap='Greens', cbar=False)
plt.title('Confusion Matrix for Stacked Model')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

#Step 7: Model Evaluation

import joblib

model_filename = 'stacked_model.joblib'

joblib.dump(stacked_model, model_filename)

loaded_model = joblib.load(model_filename)

new_xyz = [[9.375, 3.0625, 1.51], [6.995, 5.125, 0.3875], [0, 3.0625, 1.93], [9.4, 3, 1.8], [9.4, 3, 1.3]]

corr_maintenance_steps = loaded_model.predict(new_xyz)

print("Predicted Maintenance Steps For New Coordinates:",corr_maintenance_steps)
print("")
print("")
