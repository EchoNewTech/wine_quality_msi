import os
import copy
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate
import shap
import matplotlib.pyplot as plt
import warnings
import e1

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, 'baza', 'winequality_combined.csv')
df = pd.read_csv(csv_path)
X_df = df.drop(columns=['quality', 'type'])
X = X_df.values
y = df['quality'].values
feature_names = X_df.columns.tolist()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y)


model_accuracies = {}
model_feature_ranks = {}

for model_name, model_instance in e1.models.items():
    print(f"\n ANALIZOWANY MODEL: {model_name}")
    model = copy.deepcopy(model_instance)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = balanced_accuracy_score(y_test, y_pred)
    print(f"Balanced Accuracy: {acc:.4f}\n")
    model_accuracies[model_name] = acc
    cm = confusion_matrix(y_test, y_pred)
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
        shap_values = explainer.shap_values(X_test)
        X_shap_display = X_test
    else:
        # shap.kmeans kompresuje cały X_train do 10 centroidów
        background = shap.kmeans(X_train, 10)
        explainer = shap.KernelExplainer(model.predict, background)
        X_shap_display = X_test[:100]
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
