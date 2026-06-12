import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import e4

results = e4.results

# polskie cechy
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

# Wino Czerwone
df_red = results["red"][['Cecha', 'Wartość SHAP (red)']].copy()
df_red.rename(columns={'Wartość SHAP (red)': 'Importance'}, inplace=True)
df_red['Wine_Type'] = 'Wino Czerwone'

# Wino Białe
df_white = results["white"][['Cecha', 'Wartość SHAP (white)']].copy()
df_white.rename(columns={'Wartość SHAP (white)': 'Importance'}, inplace=True)
df_white['Wine_Type'] = 'Wino Białe'

full_importance_df = pd.concat([df_red, df_white])

# swap na polskie cechy
full_importance_df['Cecha'] = full_importance_df['Cecha'].map(feature_translations).fillna(full_importance_df['Cecha'])

order_of_features = full_importance_df.groupby('Cecha')['Importance'].sum().sort_values(ascending=False).index

plt.figure(figsize=(12, 8))

plt.rcParams.update({
    'text.color': 'white',
    'axes.labelcolor': 'white',
    'xtick.color': 'white',
    'ytick.color': 'white',
    'axes.titlecolor': 'white'
})

ax = sns.barplot(
    data=full_importance_df, 
    x='Importance', 
    y='Cecha', 
    hue='Wine_Type', 
    order=order_of_features,
    palette=['#E63946', '#F9E79F']
)

plt.title('Porównanie istotności parametrów dla win czerwonych i białych', fontsize=16, pad=15)
plt.xlabel('Ważność cechy', fontsize=12)

# legenda
plt.legend(title='', loc='lower right', frameon=False, fontsize=11)

plt.grid(axis='x', linestyle='--', alpha=0.3)
ax.set_axisbelow(True)
ax.spines['bottom'].set_color('white')

# Usunięcie ramki
sns.despine(left=True, bottom=False)

plt.tight_layout()

plt.savefig('Istotnosc_cech_shap.png', transparent=True, dpi=300)
#                                               ^przezroczyste tło
plt.close()

print("Zapisano wykres")
