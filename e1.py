import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tabulate import tabulate

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, 'baza', 'winequality_combined.csv')
df = pd.read_csv(csv_path)

# Zmienną jest 'quality'
X = df.drop(columns=['quality']).values
y = df['quality'].values

# Normalizacja cech
scaler = StandardScaler()
X = scaler.fit_transform(X)

print(f"Wymiary cech X: {X.shape}")
print(f"Wymiary etykiet y: {y.shape}")

models = {
    "Random Forest": RandomForestClassifier(class_weight='balanced'),
    "SVM": SVC(class_weight='balanced', probability=True),
    "kNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(class_weight='balanced'),
    "GNB": GaussianNB()
}

n_splits = 2
n_repeats = 5 
total_folds = n_splits * n_repeats

results_acc = np.zeros((total_folds, len(models)))
results_f1 = np.zeros((total_folds, len(models)))
results_prec = np.zeros((total_folds, len(models)))
results_rec = np.zeros((total_folds, len(models)))
conf_matrices = {name: np.zeros((len(np.unique(y)), len(np.unique(y))), dtype=int) for name in models.keys()}

cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats)

for fold_id, (train_idx, test_idx) in enumerate(cv.split(X, y)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    for model_id, (model_name, model) in enumerate(models.items()):
        clf = model.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        # Obliczanie metryk
        accuracy = balanced_accuracy_score(y_test, y_pred)
        results_acc[fold_id, model_id] = accuracy

        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        results_f1[fold_id, model_id] = f1

        precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
        results_prec[fold_id, model_id] = precision

        recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
        results_rec[fold_id, model_id] = recall

        conf_matrices[model_name] += confusion_matrix(y_test, y_pred)

table_data = []
for model_id, model_name in enumerate(models.keys()):
    mean_acc = np.mean(results_acc[:, model_id])
    mean_f1 = np.mean(results_f1[:, model_id])
    mean_prec = np.mean(results_prec[:, model_id])
    mean_rec = np.mean(results_rec[:, model_id])
    std_acc = np.std(results_acc[:, model_id])
    
    table_data.append([
        model_name, 
        f"{mean_acc:.4f} (± {std_acc:.4f})", 
        f"{mean_f1:.4f}", 
        f"{mean_prec:.4f}", 
        f"{mean_rec:.4f}", 
        conf_matrices[model_name]
    ])

print("\n--- Tabela Podsumowująca Wyniki ---")
print(tabulate(table_data, headers=["Model", "Balanced Accuracy", "F1 Score", "Precision", "Recall", "Confusion Matrix"], tablefmt="grid"))