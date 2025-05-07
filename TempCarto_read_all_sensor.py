import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt

# Charger les données
file_path = r"C:\Users\GAYRARD\Desktop\Python\Carto\data\01_data.csv"
with open(file_path, encoding='utf-16') as f:
    lines = f.readlines()
for i, line in enumerate(lines):
    if line.strip().startswith("Balayage"):
        header_index = i
        break
data = ''.join(lines[header_index:])
df = pd.read_csv(StringIO(data), sep=';', on_bad_lines='skip')

# Conversion propre des colonnes numériques
for col in df.columns[2:]:
    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')

# Supprimer les lignes avec NaN dans les colonnes capteurs
df.dropna(subset=df.columns[2:], inplace=True)

# Conversion de la colonne Temps en datetime
df['Temps_dt'] = pd.to_datetime(df['Temps'], format='%d/%m/%Y %H:%M:%S')

# Calcul de la fréquence moyenne d’échantillonnage
time_deltas = df['Temps_dt'].diff().dt.total_seconds().dropna()
mean_sampling_interval = time_deltas.mean()
window_length = int(round(30 * 60 / mean_sampling_interval))  # nombre de points pour 30 minutes

# Fonction de détection des plateaux
def detect_plateaus(time_series, temp_series, std_window=30, std_thresh=0.1, diff_thresh=5, min_plateau_size=20):
    rolling_std = temp_series.rolling(window=std_window).std()
    rolling_mean = temp_series.rolling(window=std_window).mean()
    relative_std = (rolling_std / rolling_mean) * 100
    mask = relative_std <= std_thresh

    cleaned_data = temp_series[mask]
    cleaned_time = time_series[mask]

    temp_diff = cleaned_data.diff()
    plateau_mask = abs(temp_diff) < diff_thresh

    plateaus = []
    current_plateau = []

    for i, (is_plateau, temp) in enumerate(zip(plateau_mask, cleaned_data)):
        if is_plateau:
            current_plateau.append((cleaned_time.iloc[i], temp))
        elif len(current_plateau) >= min_plateau_size:
            plateaus.append(current_plateau)
            current_plateau = []
        else:
            current_plateau = []
    if len(current_plateau) >= min_plateau_size:
        plateaus.append(current_plateau)

    return plateaus

# Créer la figure
plt.figure(figsize=(12, 6))
best_windows = []

# Parcourir les colonnes capteurs
for col in df.columns[2:]:
    if not pd.api.types.is_numeric_dtype(df[col]):
        continue  # Ignore les colonnes non numériques

    plt.plot(df['Temps'], df[col], color='lightsteelblue')

    plateaus = detect_plateaus(df['Temps'], df[col])
    for idx, plateau in enumerate(plateaus):
        if len(plateau) < window_length:
            continue

        temps, valeurs = zip(*plateau)
        valeurs = pd.Series(valeurs)
        rolling_std = valeurs.rolling(window=window_length).std()
        min_std_idx = rolling_std.idxmin()
        if pd.isna(min_std_idx):
            continue

        start = max(0, min_std_idx - window_length + 1)
        end = start + window_length
        best_window = plateau[start:end]
        best_windows.append((col, idx, best_window))

        if best_window:
            t_vals, v_vals = zip(*best_window)
            plt.plot(t_vals, v_vals, color='gold', linewidth=6, alpha=0.7, label=f'Best 30min Plateau {col} #{idx+1}')
            v_vals_series = pd.Series(v_vals)
            #print(f"Best 30min Plateau {col} #{idx+1}: std={v_vals_series.std():.4f}, min={v_vals_series.min():.2f}, max={v_vals_series.max():.2f}")

        # Tracé plateau brut
        t_all, v_all = zip(*plateau)
        plt.plot(t_all, v_all, linewidth=2)

plt.title('Détermination des tous les paliers de température')
plt.xlabel('Temps')
plt.ylabel('Température (°C)')
plt.xticks(range(0, len(df['Temps']), 120))
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(True)
fig = plt.gcf()
fig.patch.set_linewidth(2)
fig.patch.set_edgecolor('black')
plt.show()
