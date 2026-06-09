import os
import copy
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
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

print(f"\nNajlepszy model: {BEST_MODEL_NAME} (Balanced Accuracy: {e1.ranking[0][1]:.4f})")

# Inicjalizacja walidacji krzyżowej
rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5)

X_color_df = df.drop(columns=['quality', 'type'])
X_color = X_color_df.values
y_color = df['type'].values

color_accuracies = []
color_classes = np.unique(y_color)
color_conf_matrix = np.zeros((len(color_classes), len(color_classes)), dtype=int)

for train_index, test_index in rskf.split(X_color, y_color):
    X_train_color, X_test_color = X_color[train_index], X_color[test_index]
    y_train_color, y_test_color = y_color[train_index], y_color[test_index]

    scaler_color = StandardScaler()
    X_train_color = scaler_color.fit_transform(X_train_color)
    X_test_color = scaler_color.transform(X_test_color)

    model_color = copy.deepcopy(best_model_template)
    model_color.fit(X_train_color, y_train_color)
    y_pred_color = model_color.predict(X_test_color)

    color_accuracies.append(balanced_accuracy_score(y_test_color, y_pred_color))
    color_conf_matrix += confusion_matrix(y_test_color, y_pred_color, labels=color_classes)

acc_color_mean = np.mean(color_accuracies)
print(f"\nŚrednia skuteczność modelu (Balanced Accuracy) dla koloru po CV: {acc_color_mean:.4f}")
print("Suma macierzy pomyłek dla koloru: \n" + str(color_conf_matrix))

wine_names = {1: "red", 0: "white"}
wine_types = [1, 0]  # 1 - czerwone, 0 - białe
results = {}

# LISTA NA ZBIERANIE WYNIKÓW DO KOŃCOWEJ TABELI PODSUMOWUJĄCEJ
summary_data = []

for wine_type in wine_types:
    print(f"\nAnaliza dla wina: {wine_names[wine_type]} (type={wine_type})")

    type_mask = (df['type'] == wine_type)

    X_type_df = df[type_mask].drop(columns=['quality', 'type'])
    X_type = X_type_df.values
    y_type = df[type_mask]['quality'].values

    type_accuracies = []
    type_classes = np.unique(y_type)
    type_conf_matrix = np.zeros((len(type_classes), len(type_classes)), dtype=int)

    shap_abs_means_list = []
    feature_names = X_type_df.columns.tolist()

    top_5_counts = []

    for train_index, test_index in rskf.split(X_type, y_type):
        X_train_type, X_test_type = X_type[train_index], X_type[test_index]
        y_train_type, y_test_type = y_type[train_index], y_type[test_index]

        scaler_type = StandardScaler()
        X_train_type = scaler_type.fit_transform(X_train_type)
        X_test_type = scaler_type.transform(X_test_type)

        model_type = copy.deepcopy(best_model_template)
        model_type.fit(X_train_type, y_train_type)
        y_pred_type = model_type.predict(X_test_type)

        type_accuracies.append(balanced_accuracy_score(y_test_type, y_pred_type))
        type_conf_matrix += confusion_matrix(y_test_type, y_pred_type, labels=type_classes)

        # Wyodrębnienie top 5% najlepiej ocenianych win ze zbioru testowego z obecnego foldu
        n_top = int(len(y_test_type) * 0.05)
        sorted_indices = np.argsort(y_test_type)[::-1]
        top_5_indices = sorted_indices[:n_top]

        final_indices = [idx for idx in top_5_indices if y_test_type[idx] >= 7]

        X_test_top_5 = X_test_type[final_indices]
        y_test_top_5 = y_test_type[final_indices]

        # Jeśli przez percentyl maska byłaby pusta
        if len(X_test_top_5) == 0:
            continue

        # SHAP na wyselekcjonowanej próbie 5%
        if BEST_MODEL_NAME in ["Random Forest", "Decision Tree"]:
            explainer = shap.TreeExplainer(model_type)
            shap_values = explainer.shap_values(X_test_top_5)
        else:
            background = shap.kmeans(X_train_type, 10)
            explainer = shap.KernelExplainer(model_type.predict, background)
            X_shap_display = X_test_top_5[:100]
            shap_values = explainer.shap_values(X_shap_display)

        if isinstance(shap_values, list):
            shap_abs_mean = np.mean([np.abs(sv).mean(0) for sv in shap_values], axis=0)
        elif len(np.shape(shap_values)) == 3:
            shap_abs_mean = np.mean(np.abs(shap_values), axis=(0, 2))
        else:
            shap_abs_mean = np.abs(shap_values).mean(0)

        shap_abs_means_list.append(shap_abs_mean)

    acc_type_mean = np.mean(type_accuracies)
    avg_top_5_count = np.mean(top_5_counts)

    print(f"Średnia skuteczność modelu (Balanced Accuracy) dla jakości wina {wine_names[wine_type]}: {acc_type_mean:.4f}")
    print(f"Suma macierzy pomyłek dla jakości wina {wine_names[wine_type]}: \n{type_conf_matrix}")
    print(f"Średnia liczba analizowanych obserwacji TOP 5% na fold: {avg_top_5_count:.1f}")
    print(f"\nWpływ cech na ocenę wina {wine_names[wine_type]} (uśrednione dla TOP 5% z CV):")

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

    print(tabulate(importance_df, headers='keys', tablefmt='grid', showindex=False))

    n_top_all = int(len(y_type) * 0.05)
    sorted_idx_all = np.argsort(y_type)[::-1]
    top_5_idx_all = sorted_idx_all[:n_top_all]
    
    # Wybranie z wyciętych top 5% tylko win wyższych niż 7
    final_idx_all = [idx for idx in top_5_idx_all if y_type[idx] >= 7]
    df_top_5_raw = X_type_df.iloc[final_idx_all]

    # Tworzenie tabeli z recepturą na "idealne" wino na podstawie średnich i mediany cech dla top 5%
    recipe_df = pd.DataFrame({
        'Parametr chemiczny': feature_names,
        'Średnia (Mean)': df_top_5_raw.mean().values.round(3),
        'Mediana (Median)': df_top_5_raw.median().values.round(3)
    })

    # Dopasowanie nazwy kolumny "Cecha" do złączenia z importance_df
    importance_subset = importance_df[[nazwa_kolumny_rangi, 'Cecha']].rename(columns={'Cecha': 'Parametr chemiczny'})
    recipe_df = pd.merge(importance_subset, recipe_df, on='Parametr chemiczny')
    recipe_df = recipe_df.sort_values(by=nazwa_kolumny_rangi)

    print(f"\n--- RECEPTURA NA 'IDEALNE' WINO {wine_names[wine_type].upper()} (na podstawie najlepszych {len(df_top_5_raw)} win) ---")
    
    # Wyświetlenie tylko wybranych kolumn w tabeli
    columns_to_display = ['Parametr chemiczny', 'Średnia (Mean)', 'Mediana (Median)']
    print(tabulate(recipe_df[columns_to_display], headers='keys', tablefmt='grid', showindex=False))

    # Przygotowanie danych do predykcji jakości na podstawie średniej i mediany cech dla top 5%
    final_scaler = StandardScaler()
    X_type_scaled = final_scaler.fit_transform(X_type)
    
    final_model = copy.deepcopy(best_model_template)
    try:
        final_model.set_params(class_weight='balanced')
    except ValueError:
        pass

    final_model.fit(X_type_scaled, y_type)

    mean_vector = df_top_5_raw.mean().values.reshape(1, -1)
    median_vector = df_top_5_raw.median().values.reshape(1, -1)

    pred_mean = final_model.predict(final_scaler.transform(mean_vector))[0]
    pred_median = final_model.predict(final_scaler.transform(median_vector))[0]

    # ZBIERANIE WYNIKÓW DO TABELI PODSUMOWUJĄCEJ
    summary_data.append({
        'Typ wina': wine_names[wine_type].upper(),
        'Ocena dla przepisu ŚREDNIEJ (Mean)': int(pred_mean),
        'Ocena dla przepisu MEDIANY (Median)': int(pred_median)
    })

print("\nPORÓWNANIE WAŻNOŚCI CECH DLA TOP 5% WIN (Walidacja Krzyżowa)")
if "red" in results and "white" in results:
    df_red = results["red"][['Cecha', 'Ranga (red)']].set_index('Cecha')
    df_white = results["white"][['Cecha', 'Ranga (white)']].set_index('Cecha')
    comparison_df = df_red.join(df_white)
    comparison_df['Różnica pozycji'] = np.abs(comparison_df['Ranga (red)'] - comparison_df['Ranga (white)'])
    comparison_df = comparison_df.sort_values(by='Różnica pozycji', ascending=False).reset_index()
    print(tabulate(comparison_df, headers='keys', tablefmt='grid', showindex=False))
else:
    print("Brak pełnych danych do wygenerowania porównania.")

if summary_data:
    print("\nPODSUMOWANIE OCEN MODELU DLA RECEPTUR:")
    summary_df = pd.DataFrame(summary_data)
    print(tabulate(summary_df, headers='keys', tablefmt='grid', showindex=False))
