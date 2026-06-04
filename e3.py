import e1
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

threshold= 7

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, 'baza', 'winequality_combined.csv')
df = pd.read_csv(csv_path)

df['is_premium'] = (df['quality'] >= threshold).astype(bool)

X3 = df.drop(columns=['quality', 'is_premium','type'])
y3 = df['is_premium']

scaler = StandardScaler()
X_scaled3 = scaler.fit_transform(X3)

model = e1.models[e1.ranking[0][0]]
scores = cross_val_score(model, X_scaled3, y3, cv=10, scoring='accuracy')

print(f"--- Eksperyment Klasyfikacji Binarnej (Próg: {threshold}) ---")
print(f"Liczebność klas: \n{df['is_premium'].value_counts(normalize=True)}")
print(f"Średnia dokładność: {scores.mean():.4f} (+/- {scores.std():.4f})")