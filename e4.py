import os
import copy
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, recall_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold 
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate
import e1
import shap

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, 'baza', 'winequality_combined.csv')
df = pd.read_csv(csv_path)

BEST_MODEL_NAME = e1.ranking[0][0]
best_model_template = e1.models[BEST_MODEL_NAME]

print(f"\nNajlepszy model: {BEST_MODEL_NAME}")

rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5)

X_color_df = df.drop(columns=['quality', 'type'])
X_color = X_color_df.values
y_color = df['type'].values

color_accuracies = []
# Zbieranie unikalnych klas do sumowania macierzy pomyłek
color_classes = np.unique(y_color)
color_conf_matrix = np.zeros((len(color_classes), len(color_classes)), dtype=int)

for train_index, test_index in rskf.split(X_color, y_color):
    X_train_c, X_test_c = X_color[train_index], X_color[test_index]
    y_train_c, y_test_c = y_color[train_index], y_color[test_index]

    scaler_color = StandardScaler()
    X_train_c = scaler_color.fit_transform(X_train_c)
    X_test_c = scaler_color.transform(X_test_c)

    model_color = copy.deepcopy(best_model_template)
    model_color.fit(X_train_c, y_train_c)
    y_pred_c = model_color.predict(X_test_c)

    color_accuracies.append(balanced_accuracy_score(y_test_c, y_pred_c))
    color_conf_matrix += confusion_matrix(y_test_c, y_pred_c, labels=color_classes)

acc_color_mean = np.mean(color_accuracies)
print(f"\nŚrednia skuteczność modelu (Balanced Accuracy) dla koloru po CV (2 splits, 5 repeats): {acc_color_mean:.4f}")
print("Suma macierzy pomyłek dla koloru: \n" + str(color_conf_matrix))

wine_names = {1: "red", 0: "white"}
wine_types = [1, 0]  # 1 - czerwone, 0 - białe
results = {}

for wine_type in wine_types:
    print(f"\nAnaliza dla wina: {wine_names[wine_type]} (type={wine_type})")

    type_mask = (df['type'] == wine_type)

    X_type_df = df[type_mask].drop(columns=['quality', 'type'])
    X_type = X_type_df.values
    y_type = df[type_mask]['quality'].values

    type_accuracies = []
    type_recalls = []
    type_f1s = []
    
    type_classes = np.unique(y_type)
    type_conf_matrix = np.zeros((len(type_classes), len(type_classes)), dtype=int)

    shap_abs_means_list = []
    feature_names = X_type_df.columns.tolist()

    for train_index, test_index in rskf.split(X_type, y_type):
        X_train_t, X_test_t = X_type[train_index], X_type[test_index]
        y_train_t, y_test_t = y_type[train_index], y_type[test_index]

        scaler_type = StandardScaler()
        X_train_t = scaler_type.fit_transform(X_train_t)
        X_test_t = scaler_type.transform(X_test_t)

        model_type = copy.deepcopy(best_model_template)
        model_type.fit(X_train_t, y_train_t)
        y_pred_t = model_type.predict(X_test_t)

        # Zbieranie nowych metryk (z uwzględnieniem problemu wieloklasowego)
        type_accuracies.append(balanced_accuracy_score(y_test_t, y_pred_t))
        type_recalls.append(recall_score(y_test_t, y_pred_t, average='macro', zero_division=0))
        type_f1s.append(f1_score(y_test_t, y_pred_t, average='macro', zero_division=0))
        
        type_conf_matrix += confusion_matrix(y_test_t, y_pred_t, labels=type_classes)

        # Analiza SHAP dla bieżącego foldu
        if BEST_MODEL_NAME in ["Random Forest", "Decision Tree"]:
            explainer = shap.TreeExplainer(model_type)
            shap_values = explainer.shap_values(X_test_t)
        else:
            background = shap.kmeans(X_train_t, 10)
            explainer = shap.KernelExplainer(model_type.predict, background)
            shap_values = explainer.shap_values(X_test_t[:100])

        if isinstance(shap_values, list):
            shap_abs_mean = np.mean([np.abs(sv).mean(0) for sv in shap_values], axis=0)
        elif len(np.shape(shap_values)) == 3:
            shap_abs_mean = np.mean(np.abs(shap_values), axis=(0, 2))
        else:
            shap_abs_mean = np.abs(shap_values).mean(0)

        shap_abs_means_list.append(shap_abs_mean)

    acc_type_mean = np.mean(type_accuracies)
    print(f"Średnia skuteczność (Balanced Acc) dla jakości wina {wine_names[wine_type]}: {acc_type_mean:.4f}")
    print(f"Suma macierzy pomyłek dla jakości wina {wine_names[wine_type]}: \n{type_conf_matrix}")

    # Uśrednianie wpływu cech (SHAP) ze wszystkich iteracji
    print(f"\nŚredni wpływ cech na ocenę wina {wine_names[wine_type]} (na podstawie CV):")
    final_shap_abs_mean = np.mean(shap_abs_means_list, axis=0)

    nazwa_kolumny_shap = f'Wartość SHAP ({wine_names[wine_type]})'
    nazwa_kolumny_rangi = f'Ranga ({wine_names[wine_type]})'

    importance_df = pd.DataFrame({
        'Cecha': feature_names,
        nazwa_kolumny_shap: final_shap_abs_mean
    }).sort_values(by=nazwa_kolumny_shap, ascending=False)

    importance_df[nazwa_kolumny_rangi] = importance_df[nazwa_kolumny_shap].rank(ascending=False).astype(int)
    importance_df = importance_df[[nazwa_kolumny_rangi, 'Cecha', nazwa_kolumny_shap]]
    results[wine_names[wine_type]] = importance_df

    # Wyświetlenie tabeli z ważnością cech
    print(tabulate(importance_df, headers='keys', tablefmt='grid', showindex=False))

    # --- NOWA SEKCJA: Podsumowanie metryk klasyfikatora ---
    mean_acc = np.mean(type_accuracies)
    std_acc = np.std(type_accuracies)
    
    mean_rec = np.mean(type_recalls)
    std_rec = np.std(type_recalls)
    
    mean_f1 = np.mean(type_f1s)
    std_f1 = np.std(type_f1s)

    print(f"\nPodsumowanie skuteczności dla wina: {wine_names[wine_type]}")
    print(tabulate([
        ["Balanced Accuracy", f"{mean_acc:.4f}", f"± {std_acc:.4f}"],
        ["Recall", f"{mean_rec:.4f}", f"± {std_rec:.4f}"],
        ["F1 Score", f"{mean_f1:.4f}", f"± {std_f1:.4f}"]
    ], headers=["Metryka", "Średni Wynik", "Odchylenie Stand."], tablefmt="grid"))

print("\nBEZPOŚREDNIE PORÓWNANIE WAŻNOŚCI CECH")

if "red" in results and "white" in results:
    df_red = results["red"][['Cecha', 'Ranga (red)']].set_index('Cecha')
    df_white = results["white"][['Cecha', 'Ranga (white)']].set_index('Cecha')
    comparison_df = df_red.join(df_white)
    comparison_df['Różnica pozycji'] = np.abs(comparison_df['Ranga (red)'] - comparison_df['Ranga (white)'])
    comparison_df = comparison_df.sort_values(by='Różnica pozycji', ascending=False).reset_index()
    print(tabulate(comparison_df, headers='keys', tablefmt='grid', showindex=False))
else:
    print("Brak pełnych danych do wygenerowania porównania.")
