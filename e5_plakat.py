import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import e5

# Słownik z jednostkami dla poszczególnych cech chemicznych wina
UNITS = {
    'fixed acidity': 'g/dm³', 'fixed_acidity': 'g/dm³',
    'volatile acidity': 'g/dm³', 'volatile_acidity': 'g/dm³',
    'citric acid': 'g/dm³', 'citric_acid': 'g/dm³',
    'residual sugar': 'g/dm³', 'residual_sugar': 'g/dm³',
    'chlorides': 'g/dm³',
    'free sulfur dioxide': 'mg/dm³', 'free_sulfur_dioxide': 'mg/dm³',
    'total sulfur dioxide': 'mg/dm³', 'total_sulfur_dioxide': 'mg/dm³',
    'density': 'g/cm³',
    'pH': '', 'ph': '',
    'sulphates': 'g/dm³',
    'alcohol': '% vol'
}

def get_top_5_percent_stats(wine_type_val, metric='mean'):
    """
    Oblicza wartości cech dla fizycznego 5% najlepszych win, z zastrzeżeniem,
    że bierzemy pod uwagę wyłącznie oceny >= 7.
    """
    df_wine = e5.df[e5.df['type'] == wine_type_val]
    
    # Obliczenie liczby win, które stanowią 5% zbioru
    n_top = int(len(df_wine) * 0.05)
    # Posortowanie win według jakości i wybranie top
    df_sorted = df_wine.sort_values(by='quality', ascending=False).head(n_top)
    # Zostawienie tylko tych, które mają ocenę 7 lub wyższą
    top_wines = df_sorted[df_sorted['quality'] >= 7]
    
    if metric == 'median':
        return top_wines.median(numeric_only=True)
    return top_wines.mean(numeric_only=True)

def draw_and_save_label(title, color_bg, color_text, color_accent, top_features, values, filename):
    """
    Rysuje pojedynczą pionową etykietę wina i zapisuje ją do pliku.
    """
    # Ustawienie pionowych proporcji (np. na butelkę)
    fig, ax = plt.subplots(figsize=(5, 8))
    
    # Tło całej grafiki
    fig.patch.set_facecolor(color_bg)
    ax.set_facecolor(color_bg)
    ax.axis('off')
    
    # Rysowanie ramek etykiety
    rect = patches.Rectangle((0.04, 0.03), 0.92, 0.94, linewidth=4, edgecolor=color_accent, facecolor=color_bg, zorder=1)
    ax.add_patch(rect)
    
    inner_rect = patches.Rectangle((0.07, 0.05), 0.86, 0.90, linewidth=1, edgecolor=color_accent, facecolor='none', linestyle='--', zorder=2)
    ax.add_patch(inner_rect)

    # Nagłówek etykiety
    ax.text(0.5, 0.88, "T O P   5 %   N A J L E P S Z Y C H", fontsize=10, color=color_text, ha='center', va='center', fontfamily='serif', zorder=3)
    ax.text(0.5, 0.81, title, fontsize=26, color=color_accent, ha='center', va='center', fontweight='bold', fontfamily='serif', zorder=3)
    
    # Linia oddzielająca
    ax.plot([0.2, 0.8], [0.73, 0.73], color=color_accent, lw=2, zorder=3)
    
    # Tytuł sekcji z cechami
    ax.text(0.5, 0.65, "IDEALNE PARAMETRY:", fontsize=11, color=color_text, ha='center', va='center', fontweight='bold', zorder=3)
    
    # Wypisywanie 6 cech wraz z wartościami i jednostkami
    y_pos = 0.55
    for feature in top_features:
        clean_feature = 'pH' if feature.lower() == 'ph' else feature.replace('_', ' ').title()
        val = values[feature]
        unit = UNITS.get(feature, '')
        
        # Cecha po lewej stronie
        ax.text(0.12, y_pos, clean_feature, fontsize=12, color=color_text, ha='left', va='center', fontfamily='serif', style='italic', zorder=3)
        # Wartość po prawej stronie
        ax.text(0.88, y_pos, f"{val:.2f} {unit}", fontsize=12, color=color_text, ha='right', va='center', fontfamily='serif', fontweight='bold', zorder=3)
        
        y_pos -= 0.07 # Odstęp do kolejnej linii

    # Stopka neutralna
    ax.plot([0.3, 0.7], [0.12, 0.12], color=color_accent, lw=1, zorder=3)
    ax.text(0.5, 0.07, "Oparte na modelu SHAP i statystykach\ndla najwyższej jakości", fontsize=8, color=color_text, ha='center', va='center', zorder=3)

    # Zapis pliku
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"Zapisano etykietę: {filename}")

def generate_labels():
    if "red" not in e5.results or "white" not in e5.results:
        print("Brak wyników w e5.results. Upewnij się, że plik e5.py wykonał się poprawnie.")
        return

    # Pobranie 6 najważniejszych cech
    top_red_features = e5.results["red"].sort_values(by="Ranga (red)")['Cecha'].head(6).tolist()
    top_white_features = e5.results["white"].sort_values(by="Ranga (white)")['Cecha'].head(6).tolist()

    metrics = ['mean', 'median']
    metric_labels = {'mean': 'ŚREDNIA ARYTMETYCZNA', 'median': 'MEDIANA'}

    for metric in metrics:
        print(f"\n--- Generowanie etykiet dla metryki: {metric_labels[metric]} ---")
        
        # Obliczenie statystyk dla win top 5%
        values_red = get_top_5_percent_stats(1, metric=metric)
        values_white = get_top_5_percent_stats(0, metric=metric)

        # Generowanie etykiety dla czerwonego wina
        draw_and_save_label(
            title="PREMIUM RED", 
            color_bg="#4A0E1C", 
            color_text="#FDF6E3", 
            color_accent="#D4AF37", 
            top_features=top_red_features, 
            values=values_red,
            filename=f"etykieta_czerwone_{metric}.png"
        )

        # Generowanie etykiety dla białego wina
        draw_and_save_label(
            title="PREMIUM WHITE", 
            color_bg="#F8F4E6", 
            color_text="#2F4F4F", 
            color_accent="#C5B358", 
            top_features=top_white_features, 
            values=values_white,
            filename=f"etykieta_biale_{metric}.png"
        )

if __name__ == "__main__":
    print("Generowanie etykiet...")
    generate_labels()
    print("\nZakończono")
