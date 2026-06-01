import pandas as pd
import os

# Konfiguracja ścieżek (ścieżka, typ wina: 1 - czerwone, 0 - białe)
wine_configs = [
    (r"C:\Users\gosia\Desktop\projekt\ekspPOP\baza\winequality-red-uc.csv", 1),
    (r"C:\Users\gosia\Desktop\projekt\ekspPOP\baza\winequality-white-kaggle.csv", 0),
    (r"C:\Users\gosia\Desktop\projekt\ekspPOP\baza\baza\winequality-white-uc.csv", 0),
    (r"C:\Users\gosia\Desktop\projekt\ekspPOP\baza\winequality-red.csv", 1),
    (r"C:\Users\gosia\Desktop\projekt\ekspPOP\baza\winequality-white.csv", 0)
]

# 1. Filtrowanie tylko istniejących plików (zabezpieczenie przed błędem)
valid_configs = [(path, w_type) for path, w_type in wine_configs if os.path.exists(path)]

# 2. Odpowiednik SQL: SELECT *, [w_type] AS type FROM tabela
# Funkcja .assign() dynamicznie tworzy nową kolumnę dla każdego wczytanego pliku
dfs = [
    pd.read_csv(path, delimiter=';', encoding='utf-8').assign(type=w_type)
    for path, w_type in valid_configs
]

if dfs:
    # 3. Odpowiednik SQL: UNION ALL + SELECT DISTINCT
    # Łączymy wszystko w jedną tabelę, usuwamy duplikaty i resetujemy indeks
    combined_df = (
        pd.concat(dfs, ignore_index=True)
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # Zapis wyniku do pliku CSV
    combined_df.to_csv('winequality_combined.csv', index=False)
    
    print(f"Przetworzono pomyślnie {len(dfs)} z {len(wine_configs)} plików.")
    print(f"Liczba wierszy w gotowym zbiorze: {len(combined_df)}")
    print(combined_df.info())
else:
    print("Błąd: Nie znaleziono żadnego z podanych plików.")