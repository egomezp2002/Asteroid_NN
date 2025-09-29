


import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import random
import rebound
from solar_simulation import simulate_solar_system_hourly
import time  # <-- Agregado para medir tiempo
from datetime import datetime
from multiprocessing import Pool
import os
import pickle
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
from scipy.stats import gaussian_kde

# Cargar archivo original existente
csv_name = "resultados_20000ast_dt0_001625_hasta2350.csv"  # Ajusta según tu caso
df = pd.read_csv(csv_name)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Cargar el archivo CSV
csv_name = "resultados_20000ast_dt0_001625_hasta2350.csv"
df = pd.read_csv(csv_name)

# Elegimos 2.000 asteroides aleatoriamente
df_sample = df.sample(n=5000, random_state=42)

# Extraemos los vectores de velocidad
vx = df_sample['vx_final'].values
vy = df_sample['vy_final'].values
vz = df_sample['vz_final'].values

# Normalizamos los vectores para obtener la dirección
norms = np.linalg.norm(np.stack((vx, vy, vz), axis=1), axis=1)
directions = -np.stack((vx, vy, vz), axis=1) / norms[:, np.newaxis]  # Dirección opuesta a la velocidad

# Elegimos un punto lejano en esa dirección (por ejemplo, 2 radios terrestres)
R = 6371  # Radio de la Tierra en km
distance = 2 * R
start_points = directions * distance  # Puntos desde donde se vería venir el asteroide

# Creamos el plot 3D
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Dibujamos la Tierra como esfera
u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:100j]
x = R * np.cos(u) * np.sin(v)
y = R * np.sin(u) * np.sin(v)
z = R * np.cos(v)
ax.plot_surface(x, y, z, color='blue', alpha=0.3)

# Dibujamos las trayectorias
for i in range(2000):
    ax.plot(
        [start_points[i, 0], 0],
        [start_points[i, 1], 0],
        [start_points[i, 2], 0],
        color='red',
        alpha=0.2
    )

# Configuración del gráfico
ax.set_title("Asteroids impacting the earth (5000 elements)", fontsize=20, fontweight='bold')
ax.set_xlabel("X", fontsize=16, fontweight='bold')
ax.set_ylabel("Y", fontsize=16, fontweight='bold')
ax.set_zlabel("Z", fontsize=16, fontweight='bold')
ax.set_xlim(-distance, distance)
ax.set_ylim(-distance, distance)
ax.set_zlim(-distance, distance)
ax.set_box_aspect([1,1,1])

plt.show()

# Calculamos la dirección normalizada inversa (hacia la Tierra)
# Dirección de llegada (hacia el origen)
velocities = np.stack((vx, vy, vz), axis=1)
directions = -velocities / np.linalg.norm(velocities, axis=1)[:, np.newaxis]

# Radio de la Tierra
R = 6371  # km
impact_points = directions * R
x, y, z = impact_points[:, 0], impact_points[:, 1], impact_points[:, 2]

# Latitud (de -90 a 90)
latitudes = np.degrees(np.arcsin(z / R))

# Azimut (ángulo en plano XY, de 0 a 360)
azimuths = np.degrees(np.arctan2(y, x)) % 360

# Construcción del heatmap: latitud vs azimut
azi_bins = np.linspace(0, 360, 200)
lat_bins = np.linspace(-90, 90, 100)

# Crear bins de latitud
lat_bins = np.linspace(-90, 90, 100)

# Calcular el histograma 1D de impactos por latitud
density, lat_edges = np.histogram(latitudes, bins=lat_bins)

# Calcular centro de cada bin para graficar
lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])

# Graficar densidad vs latitud
plt.figure(figsize=(10, 5))
plt.plot(lat_centers, density, color='darkblue', linewidth=2)
plt.fill_between(lat_centers, density, alpha=0.3, color='skyblue')
plt.title("Impact Density vs Latitude", fontsize=20, fontweight='bold')
plt.xlabel("Latitude (°)", fontsize=16, fontweight='bold')
plt.ylabel("Number of Impacts", fontsize=16, fontweight='bold')
plt.grid(True)
plt.tight_layout()
plt.show()
# Contar cuántos tienen a > 0
a_mayor_que_cero = (df['a'] > 0).sum()

# Contar cuántos tienen a < 0
a_menor_que_cero = (df['a'] < 0).sum()

print(f"a > 0: {a_mayor_que_cero}")
print(f"a < 0: {a_menor_que_cero}")

# --------------------------------------------------------------------------------------------#
# OPCIÓN 1

df_no_collision = df.copy()

# Evitar modificar la columna 'label' si existe
columns_to_multiply = df_no_collision.columns.difference(['label'])

# Aplicar el escalamiento aleatorio
df_no_collision[columns_to_multiply] *= np.random.uniform(0.10, 1.90, size=df_no_collision[columns_to_multiply].shape)

# Etiquetar como no colisión
df_no_collision["label"] = 0

# Asegurar que el original tenga label = 1
df["label"] = 1



# -----------------------------------------
# 🔁 Añadir 2000 asteroides que NO chocan - OPCIÓN 2
# -----------------------------------------
# num_samples = 2000
# frac_hiperbolicos = 0.5
# num_hiper = int(num_samples * frac_hiperbolicos)
# num_elip = num_samples - num_hiper

# # ----------------------
# # 1. ÓRBITAS ELÍPTICAS
# # ----------------------
# a_elip = np.random.uniform(0.5, 15.2, num_elip)         # a > 0
# e_elip = np.random.uniform(0.0, 0.95, num_elip)        # e < 1

# # ----------------------
# # 2. ÓRBITAS HIPERBÓLICAS
# # ----------------------
# a_hiper = np.random.uniform(-15.2, -0.5, num_hiper)     # a < 0
# e_hiper = np.random.uniform(1.01, 2.0, num_hiper)      # e > 1

# # ----------------------
# # 3. Resto de parámetros comunes
# # ----------------------
# i_vals = np.random.uniform(0, 40, num_samples)
# Omega_vals = np.random.uniform(0, 360, num_samples)
# omega_vals = np.random.uniform(0, 360, num_samples)

# # ----------------------
# # 4. Combinar todos
# # ----------------------
# a_vals = np.concatenate([a_elip, a_hiper])
# e_vals = np.concatenate([e_elip, e_hiper])

# # ----------------------
# # 5. Construcción del DataFrame
# # ----------------------
# synthetic_rows = []
# errores = 0

# for a, e, i, O, w in zip(a_vals, e_vals, i_vals, Omega_vals, omega_vals):
#     try:
#         # Validar consistencia física antes de guardar
#         if (a > 0 and e >= 1) or (a < 0 and e <= 1):
#             raise ValueError("Órbita físicamente inconsistente")

#         # mean_speed solo tiene sentido si la órbita es cerrada
#         n = np.sqrt(1.0 / a**3) if a > 0 else np.nan

#         # Momento angular específico (solo si órbita es cerrada)
#         h = np.sqrt(a * (1 - e**2)) if (a > 0 and e < 1) else np.nan

#         synthetic_rows.append({
#             'a': a,
#             'e': e,
#             'i': i,
#             'Omega': O,
#             'omega': w,
#             'n': n,
#             'h': h,
#             'label': 0,
#         })
#     except ValueError:
#         errores += 1
#         continue

# print(f"✅ Generados {len(synthetic_rows)} asteroides sintéticos válidos (descartados: {errores})")

# # Crear DataFrame con asteroides que no colisionan
# df_no_collision = pd.DataFrame(synthetic_rows)

#--------------------------------------------------------------------------------------------#

# Combinar con el original (suponiendo que se llama df y ya tiene 'label' = 1 para los que chocan)
df_combined = pd.concat([df, df_no_collision], ignore_index=True)

# Guardar el resultado
combined_output = "asteroides_colisiones_y_no_colisiones.csv"
df_combined.to_csv(combined_output, index=False)
print(f"✅ Dataset combinado guardado como: {combined_output}")

# Cargar el dataset combinado
df = pd.read_csv("asteroides_colisiones_y_no_colisiones.csv")

# Limitar valores extremos para un gráfico más legible
a_min, a_max = df["a"].quantile([0.01, 0.99])
e_min, e_max = df["e"].quantile([0.01, 0.99])
e_max = max(e_max, 1.05)  # asegúrate de que e=1 entre en la gráfica

# Crear el scatter plot
plt.figure(figsize=(10, 6))
colors = {0: "tab:blue", 1: "tab:red"}

for label in [0, 1]:
    subset = df[df["label"] == label]
    plt.scatter(subset["a"], subset["e"], s=10, alpha=0.6, label=f"Label {label}", color=colors[label])

plt.axhline(y=1.0, color="black", linestyle="--", linewidth=1.5, label="e = 1 (Limit)")

plt.xlim(a_min, a_max)
plt.ylim(e_min, e_max)
plt.xlabel("Semimajor axis (AU)", fontsize=16, fontweight='bold')
plt.ylabel("Excentricity e", fontsize=16, fontweight='bold')
plt.title("A vs e: Impactors and non-impactors", fontsize=20, fontweight='bold')
plt.suptitle("1 = Impactor, 0 = Non-impactor", fontsize=16, fontweight='bold')

leg = plt.legend(fontsize=14)
for txt in leg.get_texts():
    txt.set_fontweight('bold')

plt.grid(True)
plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

parametros = ['i', 'n', 'h']
colores = {0: 'tab:blue', 1: 'tab:red'}

for param in parametros:
    plt.figure(figsize=(10, 10))

    # Extraer datos y quitar NaNs
    data0 = df[df['label'] == 0][param].dropna()
    data1 = df[df['label'] == 1][param].dropna()

    # Filtrar percentiles solo si el parámetro es 'n' o 'h'
    if param in ['n', 'h']:
        p10 = np.percentile(pd.concat([data0, data1]), 10)
        p90 = np.percentile(pd.concat([data0, data1]), 90)
        data0 = data0[(data0 >= p10) & (data0 <= p90)]
        data1 = data1[(data1 >= p10) & (data1 <= p90)]
    # Para todos los casos: definir el rango para KDE y ejes
    min_val = min(data0.min(), data1.min())
    max_val = max(data0.max(), data1.max())
    x_vals = np.linspace(min_val, max_val, 500)

    # Histograma
    plt.hist(data0, bins=50, alpha=0.4, color=colores[0], label='Non-impactor (0)', density=True, edgecolor='black')
    plt.hist(data1, bins=50, alpha=0.4, color=colores[1], label='Impactor (1)', density=True, edgecolor='black')

    # KDE
    if len(data0) > 1:
        kde0 = gaussian_kde(data0)
        plt.plot(x_vals, kde0(x_vals), color=colores[0], lw=3, label='KDE Non-impactor')
    if len(data1) > 1:
        kde1 = gaussian_kde(data1)
        plt.plot(x_vals, kde1(x_vals), color=colores[1], lw=3, label='KDE Impactor')

# Títulos dinámicos
    title_txt = (f"Distribution of '{param}' — Impactors and Non-impactors (central 100% of dataset)"
             if param in ['n', 'h'] else f"Distribution of '{param}'")

# Estilo mejorado
    plt.title(title_txt, fontsize=20, fontweight='bold')
    plt.xlabel(param, fontsize=16, fontweight='bold')
    plt.ylabel("Density", fontsize=16, fontweight='bold')

# Leyenda destacada
    leg = plt.legend(fontsize=14)
    for text in leg.get_texts():
        text.set_fontweight('bold')

# Ejes más claros
    plt.tick_params(axis='both', labelsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()
