import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import e4

results = e4.results

# Słownik do tłumaczenia cech na język polski
feature_translations = {
    'alcohol': 'Alkohol',
    'density': 'Gęstość',
    'volatile acidity': 'Kwasowość lotna',
    'free sulfur dioxide': 'Wolny dwutlenek siarki',
    'total sulfur dioxide': 'Całk. dwutlenek siarki',
    'residual sugar': 'Cukier resztkowy',
    'chlorides': 'Chlorki',
    'citric acid': 'Kwas cytrynowy',
    'pH': 'Poziom pH',
    'sulphates': 'Siarczany',
    'fixed acidity': 'Kwasowość stała'
}

# Przygotowanie danych do formatu długiego (long format) wymaganego przez Seaborn
# Wino Czerwone
df_red = results["red"][['Cecha', 'Wartość SHAP (red)']].copy()
df_red.rename(columns={'Wartość SHAP (red)': 'Importance'}, inplace=True)
df_red['Wine_Type'] = 'Wino Czerwone'

# Wino Białe
df_white = results["white"][['Cecha', 'Wartość SHAP (white)']].copy()
df_white.rename(columns={'Wartość SHAP (white)': 'Importance'}, inplace=True)
df_white['Wine_Type'] = 'Wino Białe'

# Łączymy dane
full_importance_df = pd.concat([df_red, df_white])

# Tłumaczenie angielskich nazw na polskie
full_importance_df['Cecha'] = full_importance_df['Cecha'].map(feature_translations).fillna(full_importance_df['Cecha'])

# Sortowanie cech od najważniejszej do najmniej ważnej (wg sumy SHAP dla obu win)
order_of_features = full_importance_df.groupby('Cecha')['Importance'].sum().sort_values(ascending=False).index

plt.figure(figsize=(12, 8))

# Rysowanie wykresu słupkowego z podziałem na typ wina
ax = sns.barplot(
    data=full_importance_df, 
    x='Importance', 
    y='Cecha', 
    hue='Wine_Type', 
    order=order_of_features,
    palette=['#8b0000', '#F9E79F']
)

plt.title('Porównanie istotności parametrów (wartość SHAP) dla obu typów win', fontsize=16, pad=15)
plt.xlabel('Wpływ na predykcję (Ważność cechy)', fontsize=12)
plt.ylabel('') # Puste, bo etykiety są na osi Y

# Oczyszczenie legendy
plt.legend(title='', loc='lower right', frameon=False, fontsize=11)

# Siatka pionowa
plt.grid(axis='x', linestyle='--', alpha=0.3)
ax.set_axisbelow(True) # Siatka znajduje się pod słupkami

# Usunięcie ramki
sns.despine(left=True, bottom=False)

plt.tight_layout()

# Zapisanie wykresu z przezroczystym tłem
plt.savefig('wplyw_cech_shap_seaborn.png', transparent=True, dpi=300)
plt.close()

print("Wygenerowano plik 'wplyw_cech_shap_seaborn.png'.")
