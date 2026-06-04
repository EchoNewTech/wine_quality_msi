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
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings
import shap
import copy
from sklearn.model_selection import cross_val_score


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, 'baza', 'winequality_combined.csv')
df = pd.read_csv(csv_path)

#eksperyment 1

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

#eksperyment 2

warnings.filterwarnings('ignore')

X_df = df.drop(columns=['quality', 'type'])
X2 = X_df.values
y2 = df['quality'].values
feature_names = X_df.columns.tolist()

X_train_raw, X_test_raw, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, stratify=y)

scaler = StandardScaler()
X_train2 = scaler.fit_transform(X_train_raw)
X_test2 = scaler.transform(X_test_raw)

model_accuracies = {}
model_feature_ranks = {}

for model_name, model_instance in models.items():
    print(f"\n ANALIZOWANY MODEL: {model_name}")
    model = copy.deepcopy(model_instance)
    model.fit(X_train2, y_train2)
    y_pred2 = model.predict(X_test2)
    acc = balanced_accuracy_score(y_test2, y_pred2)
    print(f"Balanced Accuracy: {acc:.4f}\n")
    model_accuracies[model_name] = acc
    cm = confusion_matrix(y_test2, y_pred2)
    unique_classes = np.unique(y)

    print("--- Najczęstsze pomyłki (próg > 15%) ---")
    for i, true_class in enumerate(unique_classes):
        total_true = np.sum(cm[i, :])
        if total_true == 0: continue

        for j, pred_class in enumerate(unique_classes):
            if i != j and cm[i, j] > 0:
                error_rate = cm[i, j] / total_true * 100
                if error_rate > 15:
                    print(
                        f"Jakość {true_class} mylona z {pred_class}: {error_rate:.1f}% przypadków ({cm[i, j]} błędów).")

    print("\n--- Analiza SHAP ---")

    if model_name in ["Random Forest", "Decision Tree"]:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test2)
        X_shap_display = X_test2
    else:
        # shap.kmeans kompresuje cały X_train do 10 centroidów
        background = shap.kmeans(X_train2, 10)
        explainer = shap.KernelExplainer(model.predict, background)
        X_shap_display = X_test2[:100]
        shap_values = explainer.shap_values(X_shap_display)

    # Agregacja wartości SHAP do wspólnego rankingu
    if isinstance(shap_values, list):
        shap_abs_mean = np.mean([np.abs(sv).mean(0) for sv in shap_values], axis=0)
    elif len(np.shape(shap_values)) == 3:
        shap_abs_mean = np.mean(np.abs(shap_values), axis=(0, 2))
    else:
        shap_abs_mean = np.abs(shap_values).mean(0)

    importance_df = pd.DataFrame({
        'Cecha (Komponent)': feature_names,
        'Wartość wpływu (SHAP)': shap_abs_mean
    }).sort_values(by='Wartość wpływu (SHAP)', ascending=False)

    print(tabulate(importance_df, headers='keys', tablefmt='grid', showindex=False))

    importance_df['Ranga'] = importance_df['Wartość wpływu (SHAP)'].rank(ascending=False).astype(int)
    model_feature_ranks[model_name] = dict(zip(importance_df['Cecha (Komponent)'], importance_df['Ranga']))

    # Wykresy
    plt.figure()
    shap.summary_plot(shap_values, X_shap_display, feature_names=feature_names, show=False)
    plt.title(f"Wpływ cech na jakosc - {model_name}")
    plt.tight_layout()
    filename = f'shap_summary_{model_name.replace(" ", "_").lower()}.png'
    plot_path = os.path.join(BASE_DIR, filename)
    plt.savefig(plot_path)
    plt.close()

    print(f"\nWykres SHAP został zapisany w tle jako: {filename}")

acc_df = pd.DataFrame(list(model_accuracies.items()), columns=['Model', 'Balanced Accuracy'])
acc_df = acc_df.sort_values(by='Balanced Accuracy', ascending=False)
print("\n--- SKUTECZNOŚĆ MODELI ---")
print(tabulate(acc_df, headers='keys', tablefmt='grid', showindex=False))

#eksperyment 3

threshold= 7

df3=df
df3['is_premium'] = (df3['quality'] >= threshold).astype(bool)

X3 = df3.drop(columns=['quality', 'is_premium','type'])
y3 = df3['is_premium']

scaler = StandardScaler()
X_scaled3 = scaler.fit_transform(X3)

model = models[ranking[0][0]]
scores = cross_val_score(model, X_scaled3, y3, cv=10, scoring='accuracy')

print(f"--- Eksperyment Klasyfikacji Binarnej (Próg: {threshold}) ---")
print(f"Liczebność klas: \n{df3['is_premium'].value_counts(normalize=True)}")
print(f"Średnia dokładność: {scores.mean():.4f} (+/- {scores.std():.4f})")