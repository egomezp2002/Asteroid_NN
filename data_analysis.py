

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
orbitales = []
for _, row in df.iterrows():
    r = [float(row['x_final']), float(row['y_final']), float(row['z_final'])]
    v = [float(row['vx_final']), float(row['vy_final']), float(row['vz_final'])]
    elems = orbital_elements(r, v)
    orbitales.append(elems)

orb_df = pd.DataFrame(orbitales)
df = pd.concat([df, orb_df], axis=1)

# Añadir columna con el año de referencia (época orbital)
df["año_epoca"] = datetime.now().year

# Guardar el archivo ampliado (puede sobrescribir el original o guardar con nuevo nombre)
output_name = csv_name.replace(".csv", "_con_orbitales.csv")
df.to_csv(output_name, index=False)

print(f"✅ CSV ampliado con elementos orbitales guardado como: {output_name}")
