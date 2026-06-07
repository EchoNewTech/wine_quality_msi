import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import e5  # Uruchomienie Twojego skryptu analizy

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

def get_top_5_percent_means(wine_type_val):
    """
    Oblicza średnie wartości cech dla 5% najwyżej ocenianych win danego typu (1 - czerwone, 0 - białe)
    na podstawie całego zbioru danych z e5.py.
    """
    df_wine = e5.df[e5.df['type'] == wine_type_val]
    threshold = np.percentile(df_wine['quality'], 95)
    top_wines = df_wine[df_wine['quality'] >= threshold]
    return top_wines.mean()

def draw_and_save_label(title, color_bg, color_text, color_accent, top_features, means, filename):
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
        clean_feature = feature.replace('_', ' ').title()
        val = means[feature]
        unit = UNITS.get(feature, '')
        
        # Cecha po lewej stronie
        ax.text(0.12, y_pos, clean_feature, fontsize=12, color=color_text, ha='left', va='center', fontfamily='serif', style='italic', zorder=3)
        # Wartość po prawej stronie
        ax.text(0.88, y_pos, f"{val:.2f} {unit}", fontsize=12, color=color_text, ha='right', va='center', fontfamily='serif', fontweight='bold', zorder=3)
        
        y_pos -= 0.07 # Odstęp do kolejnej linii

    # Stopka
    ax.plot([0.3, 0.7], [0.12, 0.12], color=color_accent, lw=1, zorder=3)
    ax.text(0.5, 0.07, "Oparte na modelu SHAP i średnich\ndla najwyższej jakości", fontsize=8, color=color_text, ha='center', va='center', zorder=3)

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

    # Obliczenie średnich parametrów dla win top 5%
    means_red = get_top_5_percent_means(1)
    means_white = get_top_5_percent_means(0)

    # Generowanie etykiety dla czerwonego wina
    draw_and_save_label(
        title="PREMIUM RED", 
        color_bg="#4A0E1C", 
        color_text="#FDF6E3", 
        color_accent="#D4AF37", 
        top_features=top_red_features, 
        means=means_red,
        filename="etykieta_czerwone.png"
    )

    # Generowanie etykiety dla białego wina
    draw_and_save_label(
        title="PREMIUM WHITE", 
        color_bg="#F8F4E6", 
        color_text="#2F4F4F", 
        color_accent="#C5B358", 
        top_features=top_white_features, 
        means=means_white,
        filename="etykieta_biale.png"
    )

if __name__ == "__main__":
    print("Rozpoczynam generowanie etykiet...")
    generate_labels()
    print("Zakończono. Pliki są gotowe do użycia na plakacie.")
