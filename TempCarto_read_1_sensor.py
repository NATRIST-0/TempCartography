import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt

file_path = r"C:\Users\GAYRARD\Desktop\Python\Carto\data\01_data.csv"

# Lire le fichier avec encodage UTF-16
with open(file_path, encoding='utf-16') as f:
    lines = f.readlines()

# Trouver l’en-tête
for i, line in enumerate(lines):
    if line.strip().startswith("Balayage"):
        header_index = i
        break

# Lire les données utiles
data = ''.join(lines[header_index:])
df = pd.read_csv(StringIO(data), sep=';', on_bad_lines='skip')

# Convertir les colonnes numériques (virgule → point)
for col in df.columns[2:]:
    df[col] = df[col].str.replace(',', '.').astype(float)

# Calculer l’écart-type relatif glissant
window_size = 30
rolling_std = df['101 (C)'].rolling(window=window_size).std()
rolling_mean = df['101 (C)'].rolling(window=window_size).mean()
relative_std = (rolling_std / rolling_mean) * 100  # En %

# Nettoyer les données (écart-type relatif ≤ 0.1 %)
mask = relative_std <= 0.1
cleaned_data = df.loc[mask, '101 (C)']
cleaned_time = df.loc[mask, 'Temps']

# Identifier les plateaux
temp_diff = cleaned_data.diff()
plateau_mask = abs(temp_diff) < 5

plateaus = []
current_plateau = []
min_plateau_size = 20

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

# Tracer les données avec deux axes Y
fig, ax1 = plt.subplots(figsize=(12, 6))

# Axe principal : température
ax1.plot(df['Temps'], df['101 (C)'], label='Temp raw data', color='lightsteelblue', linewidth = 3)
ax1.set_xlabel('Temps')
ax1.set_ylabel('Température (°C)', color='teal')
ax1.tick_params(axis='y', labelcolor='teal')

# Axe secondaire : écart-type relatif
ax2 = ax1.twinx()
ax2.plot(df['Temps'], relative_std, label='Écart-type relatif (%)', linestyle='--', color='orange', linewidth = 3)
ax2.set_ylabel('Écart-type relatif (%)', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')

# Afficher les plateaux
colors = ['red', 'forestgreen', 'navy']
for i, plateau in enumerate(plateaus):
    time_values, temp_values = zip(*plateau)
    ax1.plot(time_values, temp_values, label=f'Plateau {i+1}', color=colors[i % len(colors)], linewidth=4)

# Détermination des 30 meilleures minutes de données dans chaque plateau (meilleures 30 minutes correspond à l'écart type le plus petit sur le plateau)
best_windows = []

# Déterminer automatiquement la longueur de la fenêtre correspondant à 30 minutes
# Conversion de la colonne 'Temps' en datetime
df['Temps_dt'] = pd.to_datetime(df['Temps'], format='%d/%m/%Y %H:%M:%S')
# Calcul de la fréquence moyenne d'échantillonnage en secondes
time_deltas = df['Temps_dt'].diff().dt.total_seconds().dropna()
mean_sampling_interval = time_deltas.mean()
# Nombre de points pour 30 minutes
window_length = int(round(30 * 60 / mean_sampling_interval))

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
    best_windows.append((idx, best_window))

    # Affichage sur le graphique
    if best_window:
        t_vals, v_vals = zip(*best_window)
        ax1.plot(t_vals, v_vals, color='gold', linewidth=6, alpha=0.7, label=f'Best 30min Plateau {idx+1}')
        # Print l'écart type, min, et max de chaque Best 30min Plateau
        v_vals_series = pd.Series(v_vals)
        print(f"Best 30min Plateau {col} #{idx+1}: std={v_vals_series.std():.4f}, min={v_vals_series.min():.2f}, max={v_vals_series.max():.2f}")

# Afficher les légendes et ajuster le tracé
# lines_1, labels_1 = ax1.get_legend_handles_labels()
# lines_2, labels_2 = ax2.get_legend_handles_labels()
# legend = ax1.legend(lines_1 + lines_2, labels_1 + labels_2, bbox_to_anchor=(1.05, 1))

ax1.set_xticks(range(0, len(df['Temps']), 120))
ax1.set_xticklabels(df['Temps'][::120], rotation=45)
ax1.grid(True)
fig.subplots_adjust(top=0.986, bottom=0.204, left=0.055, right=0.705, hspace=0.2, wspace=0.2)
# ax1.xaxis.label.set_size(16)
# ax1.yaxis.label.set_size(16)
# ax2.yaxis.label.set_size(16)
#plt.title('Température de l\'enceinte thermostatique')
# fig.patch.set_linewidth(2)
# fig.patch.set_edgecolor('black')
plt.tight_layout()
plt.show()
