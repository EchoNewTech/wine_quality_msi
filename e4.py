import os
import copy
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate
import e1

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, 'baza', 'winequality_combined.csv')
df = pd.read_csv(csv_path)

X_eval_df = df.drop(columns=['quality', 'type'])
X_eval = X_eval_df.values
y_eval = df['quality'].values

X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_eval, y_eval, test_size=0.2, stratify=y_eval)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

scaler_eval = StandardScaler()
X_train_eval = scaler_eval.fit_transform(X_train_raw)
X_test_eval = scaler_eval.transform(X_test_raw)

BEST_MODEL_NAME = e1.ranking[0][0]
best_model_template = e1.models[BEST_MODEL_NAME]

print(f"\nNajlepszy model: {BEST_MODEL_NAME} (Balanced Accuracy: {e1.ranking[0][1]:.4f})")
print("Ten model zostanie użyty w dalszych krokach.")

# Klasyfikacja koloru wina (białe/czerwone - 'type')
X_color_df = df.drop(columns=['quality', 'type'])
X_color = X_color_df.values
y_color = df['type'].values

X_train_color, X_test_color, y_train_color, y_test_color = train_test_split(X_color, y_color, test_size=0.2, stratify=y_color)

scaler_color = StandardScaler()
X_train_color = scaler_color.fit_transform(X_train_color)
X_test_color = scaler_color.transform(X_test_color)

model_color = copy.deepcopy(best_model_template)
model_color.fit(X_train_color, y_train_color)
y_pred_color = model_color.predict(X_test_color)
acc_color = balanced_accuracy_score(y_test_color, y_pred_color)

print(f"\nOgólna skuteczność modelu (Balanced Accuracy) dla koloru: {acc_color:.4f}")

print("Macierz pomyłek dla koloru: \n" + str(confusion_matrix(y_test_color, y_pred_color)))

# Analiza jakości z podziałem na kolor wina
wine_names = {1: "red", 0: "white"}
wine_types = [1, 0]  # 1 - czerwone, 0 - białe
results = {}
for wine_type in wine_types:
    print(f"\nAnaliza dla wina: {wine_names[wine_type]} (type={wine_type})")

    type_mask = (df['type'] == wine_type)

    X_type = df[type_mask].drop(columns=['quality', 'type']).values
    y_type = df[type_mask]['quality'].values

    X_train_type, X_test_type, y_train_type, y_test_type = train_test_split(X_type, y_type, test_size=0.2, stratify=y_type)

    scaler_type = StandardScaler()
    X_train_type = scaler_type.fit_transform(X_train_type)
    X_test_type = scaler_type.transform(X_test_type)

    model_type = copy.deepcopy(best_model_template)
    model_type.fit(X_train_type, y_train_type)
    y_pred_type = model_type.predict(X_test_type)
    acc_type = balanced_accuracy_score(y_test_type, y_pred_type)

    print(f"Skuteczność modelu (Balanced Accuracy) dla jakości wina {wine_names[wine_type]} (type={wine_type}): {acc_type:.4f}")
    print("Macierz pomyłek dla jakości wina {}: \n{}".format(wine_names[wine_type], confusion_matrix(y_test_type, y_pred_type)))

    # Obliczenie wpływu cech na ocenę wina
    print("\nWpływ cech na ocenę wina {}:".format(wine_names[wine_type]))
    

# Podsumowanie wyników
print("\nPodsumowanie wpływu cech na ocenę wina:")
