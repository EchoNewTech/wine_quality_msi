import e1
import os
import copy
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, f1_score, precision_score, recall_score
from tabulate import tabulate
import warnings

warnings.filterwarnings('ignore')

threshold= 7

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, 'baza', 'winequality_combined.csv')
df = pd.read_csv(csv_path)

df['is_premium'] = (df['quality'] >= threshold).astype(int)

X = df.drop(columns=['quality', 'is_premium','type']).values
y = df['is_premium'].values

model = e1.ranking[0][0]
base_model = copy.deepcopy(e1.models[model])
print("\nEksperyment 3\n")

n_splits = 2
n_repeats = 5
total_folds = n_splits * n_repeats
cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats)

results_acc = np.zeros((total_folds))
results_f1 = np.zeros((total_folds))
results_prec = np.zeros((total_folds))
results_rec = np.zeros((total_folds))
conf_matrices = np.zeros((len(np.unique(y)), len(np.unique(y))), dtype=int)

for fold_id, (train_idx, test_idx) in enumerate(cv.split(X, y)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = base_model.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    results_acc[fold_id] = balanced_accuracy_score(y_test, y_pred)
    results_f1[fold_id] = f1_score(y_test, y_pred, average='binary')
    results_prec[fold_id] = precision_score(y_test, y_pred, average='binary')
    results_rec[fold_id] = recall_score(y_test, y_pred, average='binary')

    conf_matrices += confusion_matrix(y_test, y_pred)

mean_acc, std_acc = np.mean(results_acc), np.std(results_acc)
mean_f1, std_f1 = np.mean(results_f1), np.std(results_f1)
mean_prec, std_prec = np.mean(results_prec), np.std(results_prec)
mean_rec, std_rec = np.mean(results_rec), np.std(results_rec)
avg_cm = conf_matrices // n_repeats

print(tabulate([
    ["Balanced Accuracy", f"{mean_acc:.4f}", f"± {std_acc:.4f}"],
    ["Precision (Premium)", f"{mean_prec:.4f}", f"± {std_prec:.4f}"],
    ["Recall (Premium)", f"{mean_rec:.4f}", f"± {std_rec:.4f}"],
    ["F1 Score (Premium)", f"{mean_f1:.4f}", f"± {std_f1:.4f}"]
], headers=["Metryka", "Średni Wynik", "Odchylenie Stand. (Stabilność)"], tablefmt="grid"))

print(f"Średnia macierz pomyłek: \n{avg_cm}")
