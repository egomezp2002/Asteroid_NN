

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

import pandas as pd
import numpy as np
from datetime import datetime

# Cargar archivo original existente
csv_name = "resultados_2000ast_dt0_001625_hasta2350.csv"  # Ajusta según tu caso
df = pd.read_csv(csv_name)

# Función para calcular elementos orbitales
def orbital_elements(r, v, mu=1.0):
    r = np.array(r)
    v = np.array(v)
    r_mag = np.linalg.norm(r)
    v_mag = np.linalg.norm(v)
    h = np.cross(r, v)
    h_mag = np.linalg.norm(h)

    e_vec = (np.cross(v, h) / mu) - (r / r_mag)
    e = np.linalg.norm(e_vec)
    energy = 0.5 * v_mag**2 - mu / r_mag
    a = -mu / (2 * energy)

    i = np.arccos(h[2] / h_mag)
    n = np.cross([0, 0, 1], h)
    n_mag = np.linalg.norm(n)

    if n_mag == 0:
        Omega = 0
    else:
        Omega = np.arccos(n[0] / n_mag)
        if n[1] < 0:
            Omega = 2 * np.pi - Omega

    if n_mag == 0 or e < 1e-8:
        omega = 0
    else:
        omega = np.arccos(np.dot(n, e_vec) / (n_mag * e))
        if e_vec[2] < 0:
            omega = 2 * np.pi - omega

    if e < 1e-8:
        nu = 0
    else:
        nu = np.arccos(np.dot(e_vec, r) / (e * r_mag))
        if np.dot(r, v) < 0:
            nu = 2 * np.pi - nu

    return {
        "a": a,
        "e": e,
        "i": np.degrees(i),
        "Omega": np.degrees(Omega),
        "omega": np.degrees(omega),
        "nu": np.degrees(nu)
    }

# Calcular elementos orbitales y agregarlos al DataFrame
# Calcular elementos orbitales y agregarlos al DataFrame
orbitales = []
for _, row in df.iterrows():
    r = [float(row['x_final']), float(row['y_final']), float(row['z_final'])]
    v = [float(row['vx_final']), float(row['vy_final']), float(row['vz_final'])]
    elems = orbital_elements(r, v)
    
    a = elems["a"]
    if a > 0:
        mean_speed = np.sqrt(1.0 / a**3)
    else:
        mean_speed = np.nan  # O 0.0 si prefieres

    h_vec = np.cross(r, v)
    h_mag = np.linalg.norm(h_vec)

    elems["mean_speed"] = mean_speed
    elems["h"] = h_mag
    orbitales.append(elems)

orb_df = pd.DataFrame(orbitales)
df = pd.concat([df, orb_df], axis=1)

df["año_epoca"] = datetime.now().year
df["label"] = 1

output_name = csv_name.replace(".csv", "_con_orbitales.csv")
df.to_csv(output_name, index=False)

print(f"✅ CSV ampliado con elementos orbitales guardado como: {output_name}")


import numpy as np
from datetime import datetime

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
a_elip = np.random.uniform(0.5, 5.2, num_elip)         # a > 0
e_elip = np.random.uniform(0.0, 0.95, num_elip)        # e < 1

# ----------------------
# 2. ÓRBITAS HIPERBÓLICAS
# ----------------------
a_hiper = np.random.uniform(-5.2, -0.5, num_hiper)     # a < 0
e_hiper = np.random.uniform(1.01, 200.0, num_hiper)      # e > 1

# ----------------------
# 3. Resto de parámetros comunes
# ----------------------
i_vals = np.random.uniform(0, 40, num_samples)
Omega_vals = np.random.uniform(0, 360, num_samples)
omega_vals = np.random.uniform(0, 360, num_samples)
nu_vals = np.random.uniform(-120, 120, num_samples)

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

for a, e, i, O, w, nu in zip(a_vals, e_vals, i_vals, Omega_vals, omega_vals, nu_vals):
    try:
        # Validar consistencia física antes de guardar
        if (a > 0 and e >= 1) or (a < 0 and e <= 1):
            raise ValueError("Órbita físicamente inconsistente")

        # mean_speed solo tiene sentido si la órbita es cerrada
        mean_speed = np.sqrt(1.0 / a**3) if a > 0 else np.nan

        synthetic_rows.append({
            'a': a,
            'e': e,
            'i': i,
            'Omega': O,
            'omega': w,
            'nu': nu,
            'mean_speed': mean_speed,
            'label': 0,
            'año_epoca': datetime.now().year
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