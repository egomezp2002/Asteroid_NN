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
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from tensorflow import initializers

from scipy.stats import gaussian_kde
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Paso 1: Cargar los datos
csv_name = "asteroides_colisiones_y_no_colisiones.csv"
df = pd.read_csv(csv_name)

# Paso 2: Seleccionar columnas deseadas
cols_deseadas = ['a', 'e', 'i', 'n', 'h', 'label']
df = df[cols_deseadas]

# Paso 3: Calcular percentiles 5 y 95 para 'n' y 'h'
n_p5, n_p95 = df['n'].quantile([0.0, 1])
h_p5, h_p95 = df['h'].quantile([0.0, 1])

# Paso 4: Filtrar para conservar solo el 90% significativo
df_filtrado = df[
    (df['n'] >= n_p5) & (df['n'] <= n_p95) &
    (df['h'] >= h_p5) & (df['h'] <= h_p95)
]

# Paso 5: Guardar el nuevo CSV
df_filtrado.to_csv("orbitales_filtrados_v2.csv", index=False)
print("✅ CSV guardado: orbitales_filtrados_v2.csv")

parametros = ['n', 'h']
colores = {0: 'tab:blue', 1: 'tab:red'}

for param in parametros:
    plt.figure(figsize=(10, 5))

    # Separar impactores y no impactores
    data0 = df[df['label'] == 0][param].dropna()
    data1 = df[df['label'] == 1][param].dropna()

    # Filtrado del 80% central (para n y h)
    if param in ['n', 'h']:
        combined = pd.concat([data0, data1])
        p10, p90 = np.percentile(combined, [10, 90])
        data0 = data0[(data0 >= p10) & (data0 <= p90)]
        data1 = data1[(data1 >= p10) & (data1 <= p90)]

    # Rango común para KDE
    min_val = min(data0.min(), data1.min())
    max_val = max(data0.max(), data1.max())
    x_vals = np.linspace(min_val, max_val, 500)

    # Histogramas
    plt.hist(data0, bins=50, alpha=0.4, color=colores[0], label='Non-impactor (0)', density=True)
    plt.hist(data1, bins=50, alpha=0.4, color=colores[1], label='Impactor (1)', density=True)

    # KDEs
    if len(data0) > 1:
        kde0 = gaussian_kde(data0)
        plt.plot(x_vals, kde0(x_vals), color=colores[0], lw=2)
    if len(data1) > 1:
        kde1 = gaussian_kde(data1)
        plt.plot(x_vals, kde1(x_vals), color=colores[1], lw=2)

    # Título adaptado según filtro
    titulo = f"Distribution of '{param}'"
    if param in ['n', 'h']:
        titulo += " (100%)"
    
    plt.title(titulo)
    plt.xlabel(param)
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()