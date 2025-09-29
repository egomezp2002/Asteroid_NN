

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

# Cargar archivo original existente
csv_name = "resultados_20000ast_dt0_001625_hasta2350.csv"  # Ajusta según tu caso
df = pd.read_csv(csv_name)

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
# Limitar valores extremos para un gráfico más legible
a_min, a_max = df["a"].quantile([0.01, 0.99])
e_min, e_max = df["e"].quantile([0.01, 0.99])
e_max = max(e_max, 1.05)  # asegúrate de que e=1 entre en la gráfica

# Crear el scatter plot
plt.figure(figsize=(10, 6))
colors = {0: "tab:blue", 1: "tab:red"}

for label in [0, 1]:
    subset = df[df["label"] == label]
    plt.scatter(
        subset["a"], subset["e"], 
        s=10, alpha=0.6, 
        label=f"Label {label}", 
        color=colors[label]
    )

# Añadir línea horizontal e = 1
plt.axhline(y=1.0, color="black", linestyle="--", linewidth=1.5, label="e = 1 (Limit)")

# Ajustes visuales
plt.xlim(a_min, a_max)
plt.ylim(e_min, e_max)
plt.xlabel("Semimajor axis (AU)", fontsize=16, fontweight='bold')
plt.ylabel("Eccentricity e", fontsize=16, fontweight='bold')
plt.title("A vs e: Impactors and Non-impactors", fontsize=20, fontweight='bold')
plt.suptitle("1 = Impactor, 0 = Non-impactor", fontsize=16, fontweight='bold', y=0.95)
plt.legend(fontsize=14, title="Legend", title_fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=14)  # 🔥 Aumenta tamaño de los ticks
plt.grid(True)
plt.tight_layout()
plt.show()

# Histograma por parámetros
parametros = ['i', 'Omega', 'omega']
colores = {0: 'tab:blue', 1: 'tab:red'}

for param in parametros:
    plt.figure(figsize=(10, 5))

    # Histograma de no impactadores
    plt.hist(df[df['label'] == 0][param], bins=50, alpha=0.6, label='Non-impactor (0)', color=colores[0], density=True)

    # Histograma de impactadores
    plt.hist(df[df['label'] == 1][param], bins=50, alpha=0.6, label='Impactor (1)', color=colores[1], density=True)

    plt.title(f"Distribution of '{param}' — Impactors and Non-impactors", fontsize=20, fontweight='bold')
    plt.xlabel(f"{param} (rads)", fontsize=16, fontweight='bold')
    plt.ylabel("Density of data", fontsize=16, fontweight='bold')
    plt.legend(fontsize=14, title="Legend", title_fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=14)  # 🔥 Ticks más grandes
    plt.grid(True)
    plt.tight_layout()
    plt.show()
