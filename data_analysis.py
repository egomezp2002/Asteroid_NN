

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
csv_name = "resultados_2000ast_dt0_001625_hasta2350.csv"  # Ajusta según tu caso
df = pd.read_csv(csv_name)

# -----------------------------------------
# 🔁 Añadir 2000 asteroides que NO chocan
# -----------------------------------------
num_samples = 2000
frac_hiperbolicos = 0.5
num_hiper = int(num_samples * frac_hiperbolicos)
num_elip = num_samples - num_hiper

# ----------------------
# 1. ÓRBITAS ELÍPTICAS
# ----------------------
a_elip = np.random.uniform(0.5, 15.2, num_elip)         # a > 0
e_elip = np.random.uniform(0.0, 0.95, num_elip)        # e < 1

# ----------------------
# 2. ÓRBITAS HIPERBÓLICAS
# ----------------------
a_hiper = np.random.uniform(-15.2, -0.5, num_hiper)     # a < 0
e_hiper = np.random.uniform(1.01, 600.0, num_hiper)      # e > 1

# ----------------------
# 3. Resto de parámetros comunes
# ----------------------
i_vals = np.random.uniform(0, 40, num_samples)
Omega_vals = np.random.uniform(0, 360, num_samples)
omega_vals = np.random.uniform(0, 360, num_samples)

# ----------------------
# 4. Combinar todos
# ----------------------
a_vals = np.concatenate([a_elip, a_hiper])
e_vals = np.concatenate([e_elip, e_hiper])

# ----------------------
# 5. Construcción del DataFrame
# ----------------------
synthetic_rows = []
errores = 0

for a, e, i, O, w in zip(a_vals, e_vals, i_vals, Omega_vals, omega_vals):
    try:
        # Validar consistencia física antes de guardar
        if (a > 0 and e >= 1) or (a < 0 and e <= 1):
            raise ValueError("Órbita físicamente inconsistente")

        # mean_speed solo tiene sentido si la órbita es cerrada
        n = np.sqrt(1.0 / a**3) if a > 0 else np.nan

        # Momento angular específico (solo si órbita es cerrada)
        h = np.sqrt(a * (1 - e**2)) if (a > 0 and e < 1) else np.nan

        synthetic_rows.append({
            'a': a,
            'e': e,
            'i': i,
            'Omega': O,
            'omega': w,
            'n': n,
            'h': h,
            'label': 0,
        })
    except ValueError:
        errores += 1
        continue

print(f"✅ Generados {len(synthetic_rows)} asteroides sintéticos válidos (descartados: {errores})")


# Crear DataFrame con asteroides que no colisionan
df_no_collision = pd.DataFrame(synthetic_rows)

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

# Añadir línea horizontal e = 1
plt.axhline(y=1.0, color="black", linestyle="--", linewidth=1.5, label="e = 1 (límite)")

# Ajustes visuales
plt.xlim(a_min, a_max)
plt.ylim(e_min, e_max)
plt.xlabel("Semieje mayor a (AU)")
plt.ylabel("Excentricidad e")
plt.title("Diagrama a vs e — Asteroides colisionantes vs no colisionantes")
plt.suptitle("1 = Colisiona, 0 = No colisiona")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()