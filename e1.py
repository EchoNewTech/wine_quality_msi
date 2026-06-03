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
from scipy.stats import ttest_rel
from tabulate import tabulate

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, 'baza', 'winequality_combined.csv')
df = pd.read_csv(csv_path)

# Zmienną jest 'quality'
X = df.drop(columns=['quality', 'type'], errors ='ignore').values
y = df['quality'].values


print(f"Wymiary cech X: {X.shape}")
print(f"Wymiary etykiet y: {y.shape}")

models = {
    "Random Forest": RandomForestClassifier(class_weight='balanced'),
    "SVM": SVC(class_weight='balanced', cache_size=1000),
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

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

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
print(
    tabulate(table_data, headers=["Model", "Balanced Accuracy", "F1 Score", "Precision", "Recall", "Confusion Matrix"],
             tablefmt="grid"))

# Testy statystyczne
alpha = 0.05
num_models = len(models)
clfs_names = list(models.keys())

t_stats = np.full((num_models, num_models), np.nan)
p_values = np.full((num_models, num_models), np.nan)
better_models = np.zeros((num_models, num_models), dtype=bool)
significant_differences = np.zeros((num_models, num_models), dtype=bool)

for i in range(num_models):
    for j in range(num_models):
        if i != j:
            stat, p_val = ttest_rel(results_acc[:, i], results_acc[:, j])
            t_stats[i, j], p_values[i, j] = stat, p_val
            better_models[i, j] = stat > 0
            significant_differences[i, j] = p_val < alpha

# print("\n--- Test Statystyczny t-Studenta ---")
# print(f"T-Statistics:\n{t_stats}")
# print(f"\nP-Values:\n{p_values}")
# print(f"\nBetter:\n{better_models}")
# print(f"\nSignificant:\n{significant_differences}\n")

stat_table = []
for i in range(num_models):
    for j in range(i + 1, num_models):
        mean_i, mean_j = np.mean(results_acc[:, i]), np.mean(results_acc[:, j])
        name_i, name_j = clfs_names[i], clfs_names[j]

        # Interpretacja wyników testu
        pair_name = f"{name_i} vs {name_j}"
        p_val_str = f"{p_values[i, j]:.4f}"

        if significant_differences[i, j]:
            if better_models[i, j]:
                result_str = f"{name_i} (p={p_val_str})"
            else:
                result_str = f"{name_j} (p={p_val_str})"
            significance_str = "TAK"
        else:
            result_str = f"Brak różnicy (p={p_val_str})"
            significance_str = "NIE"
        stat_table.append([pair_name, result_str, significance_str])

print("\n--- Podsumowanie T-Studenta ---")
print(tabulate(stat_table, headers=["Para Modeli", "Wynik Testu", "Istotność"], tablefmt="grid"))

print("\n--- Ranking Modeli ---")
ranking = []
for i, name in enumerate(clfs_names):
    mean_acc = np.mean(results_acc[:, i])
    ranking.append((name, mean_acc))

# Sortowanie malejąco po wyniku
ranking.sort(key=lambda x: x[1], reverse=True)

for pozycja, (model, wynik) in enumerate(ranking, start=1):
    print(f"{pozycja}. {model:<13} - Średnia: {wynik:.4f}")
