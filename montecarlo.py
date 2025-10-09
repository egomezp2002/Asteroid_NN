

import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import random
import rebound
import pandas as pd
from solar_simulation import simulate_solar_system_hourly
import time  # <-- Agregado para medir tiempo
from datetime import datetime
from multiprocessing import Pool
import os
import pickle

rebound.horizons.SSL_CONTEXT = "unverified"  

end_year = 2350
start_year = 2025
years_back = end_year - start_year
n_steps = 200000
dt = -years_back / n_steps
num_asteroides = 20000
dt_label = f"{abs(dt):.6f}".replace('.', '_')
integrator_name = 'whfast'

planet_names = ["Sun", "Mercury", "Venus", "Earth", "Mars",
                "Jupiter", "Saturn", "Uranus", "Neptune", "Satellite"]

# --- Abrir resultados y elegir fila fija ---
try:
    df = pd.read_csv("resultados_20000ast_dt0_001625_hasta2350.csv")
except:
    df = pd.DataFrame(np.load("resultados_20000ast_dt0_001625_hasta2350.npy", allow_pickle=True))

muestra = df.iloc[275]   # fila número X


# Montecarlo
# Archivo donde guardaremos la simulación base ya lista
sim_cache_file = "simulacion_base.pkl"

# --------------------------------------------------------------------------
# CREAR Y GUARDAR SIMULACIÓN BASE
# --------------------------------------------------------------------------
def crear_simulacion_base():
    print(f"⚙️ Generando nueva simulación base y guardando en caché: {sim_cache_file}")
    sim = rebound.Simulation()
    sim.units = ('AU', 'yr', 'Msun')
    sim.add(["Sun", "Mercury", "Venus", "Earth", "Mars",
             "Jupiter", "Saturn", "Uranus", "Neptune"], date=f"{start_year}-01-01")
    sim.integrator = integrator_name
    sim.dt = dt
    with open(sim_cache_file, "wb") as f:
        pickle.dump(sim, f)
    print("✅ Simulación base guardada.")

# --------------------------------------------------------------------------
# CARGAR SIMULACIÓN BASE DESDE CACHÉ
# --------------------------------------------------------------------------
def cargar_simulacion_base():
    if os.path.exists(sim_cache_file):
        try:
            with open(sim_cache_file, "rb") as f:
                sim = pickle.load(f)
            return sim
        except (EOFError, pickle.UnpicklingError):
            print("⚠️ Caché dañada. Regenerando...")
            os.remove(sim_cache_file)
            crear_simulacion_base()
            return cargar_simulacion_base()
    else:
        crear_simulacion_base()
        return cargar_simulacion_base()

# --------------------------------------------------------------------------
# MONTECARLO CON SIMULACIÓN BASE CACHEADA
# --------------------------------------------------------------------------
from multiprocessing import Pool, cpu_count

def simular_un_asteroide(args):
    """
    Ejecuta UNA simulación Monte Carlo con perturbaciones.
    Devuelve 1 si hay impacto (o encuentro cercano), 0 en caso contrario.
    """
    seed, sigma_pos, sigma_vel, distancia_check_UA = args
    np.random.seed(seed)

    sim = cargar_simulacion_base().copy()

    # Valores iniciales del asteroide
    x, y, z = muestra['x_final'], muestra['y_final'], muestra['z_final']
    vx, vy, vz = muestra['vx_final'], muestra['vy_final'], muestra['vz_final']

    # Perturbaciones aleatorias
    x  += np.random.normal(0, sigma_pos)
    y  += np.random.normal(0, sigma_pos)
    z  += np.random.normal(0, sigma_pos)
    vx += np.random.normal(0, sigma_vel)
    vy += np.random.normal(0, sigma_vel)
    vz += np.random.normal(0, sigma_vel)

    # Añadir asteroide
    sim.add(x=x, y=y, z=z, vx=vx, vy=vy, vz=vz, m=0)

    earth = sim.particles[3]
    ast   = sim.particles[-1]

    # Integración en pasos intermedios
    tiempos = np.linspace(start_year, end_year, 118625)
    for t in tiempos:
        sim.integrate(t)
        d = np.sqrt((earth.x-ast.x)**2 + (earth.y-ast.y)**2 + (earth.z-ast.z)**2)
        if d < distancia_check_UA:
            return 1  # Impacto/encuentro cercano
    return 0  # No impacto


def montecarlo_paralelo(n_iter=1000, sigma_pos=1e-3, sigma_vel=1e-3):
    """
    Versión paralelizada del Monte Carlo.
    """
    distancia_check_UA = 1e-2  # ~4 distancias Tierra-Luna

    # Creamos la lista de argumentos para cada worker
    seeds = np.arange(n_iter)
    args = [(seed, sigma_pos, sigma_vel, distancia_check_UA) for seed in seeds]

    with Pool(processes=cpu_count()) as pool:
        resultados = pool.map(simular_un_asteroide, args)

    prob = 100 * sum(resultados) / n_iter
    return prob


# --------------------------------------------------------------------------
# USO
# --------------------------------------------------------------------------
if __name__ == "__main__":
    cargar_simulacion_base()  # asegurar que existe la base

    prob = montecarlo_paralelo(n_iter=400, sigma_pos=1e-4, sigma_vel=1e-4)
    print(f"Probabilidad estimada de impacto: {prob:.2f}%")

