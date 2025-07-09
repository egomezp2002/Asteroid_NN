
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

# --------------------------------------------------------------------------
# CONFIGURACIÓN GLOBAL
# --------------------------------------------------------------------------

end_year = 2350
start_year = datetime.now().year
years_back = end_year - start_year
n_steps = 200000
dt = -years_back / n_steps
num_asteroides = 2000
dt_label = f"{abs(dt):.6f}".replace('.', '_')
integrator_name = 'whfast'

planet_names = ["Sun", "Mercury", "Venus", "Earth", "Mars",
                "Jupiter", "Saturn", "Uranus", "Neptune", "Satellite"]

# --------------------------------------------------------------------------
# SIMULACIÓN DEL SISTEMA SOLAR (opcional) - No es necesario utilizar
# --------------------------------------------------------------------------
"""
print(f"Simulando el sistema solar hora a hora desde {start_year} hasta {end_year}...")
data = simulate_solar_system_hourly(end_year=end_year)
filename_base = f"planet_positions_hourly_until_{end_year}"
np.save(f"{filename_base}.npy", data)
np.savetxt(f"{filename_base}.csv", data, delimiter=",")
print(f"✅ Datos guardados en: {filename_base}.npy y .csv")
"""

# --------------------------------------------------------------------------
# PROPAGACIÓN HACIA ATRÁS DE UN ASTEROIDE CON WHFAST (Si solo queremos probar un asteroide y animarlo)
# --------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ANIMACIÓN DE LA TRAYECTORIA HACIA ATRÁS (Sólo para un satélite)
# ------------------------------------------------------------------------------
# start_time = time.time()  # 🕒 Inicio de temporizador


# sim = rebound.Simulation()
# sim.units = ('AU', 'yr', 'Msun')
# sim.add(["Sun", "Mercury", "Venus", "Earth", "Mars",
#          "Jupiter", "Saturn", "Uranus", "Neptune"], date=f"{end_year}-01-01")

# earth = sim.particles[3]
# v_kms = random.uniform(15, 45)
# v_auyr = v_kms / 4.74047
# angle = random.uniform(0, 2*np.pi)
# vx = -v_auyr * np.cos(angle)
# vy = -v_auyr * np.sin(angle)

# sim.add(x=earth.x, y=earth.y, z=earth.z,
#         vx=earth.vx + vx, vy=earth.vy + vy, vz=earth.vz,
#         m=0.0)

# sim.integrator = "whfast"
# sim.dt = dt

# positions = np.zeros((n_steps, sim.N, 2))  # solo plano XY

# for i in range(n_steps):
#     sim.integrate(sim.t + dt)
#     for j, p in enumerate(sim.particles):
#         positions[i, j, 0] = p.x
#         positions[i, j, 1] = p.y

# end_time = time.time()  # 🕒 Fin de temporizador
# elapsed = end_time - start_time
# print(f"🕒 Tiempo de propagación: {elapsed:.2f} segundos")

# fig, ax = plt.subplots(figsize=(8, 8))
# colors = ['yellow', 'gray', 'orange', 'blue', 'red', 'brown', 'gold', 'cyan', 'blueviolet', 'black']
# lines = [ax.plot([], [], color=colors[i], lw=1)[0] for i in range(len(planet_names))]
# points = [ax.plot([], [], 'o', color=colors[i], label=planet_names[i])[0] for i in range(len(planet_names))]

# ax.set_xlim(-35, 35)
# ax.set_ylim(-35, 35)
# ax.set_xlabel("Distance (AU)")        # Unidad en eje X
# ax.set_ylabel("Distance (AU)")       # Unidad en eje Y
# ax.set_aspect('equal')
# ax.legend(loc='upper right')
# year_text = ax.text(0.5, 1.02, '', transform=ax.transAxes,
#                     ha='center', va='bottom', fontsize=14, fontweight='bold')

# def update(frame):
#     for i in range(sim.N):
#         x = positions[:frame, i, 0]
#         y = positions[:frame, i, 1]
#         lines[i].set_data(x, y)
#         points[i].set_data([positions[frame-1, i, 0]], [positions[frame-1, i, 1]])
#     current_fraction = frame / n_steps
#     current_year = end_year - current_fraction * years_back
#     year_text.set_text(f"Year: {current_year:.1f}")
#     return lines + points + [year_text]

# ani = FuncAnimation(fig, update, frames=range(10, n_steps, 10), interval=30, blit=False)

# # Guardar como archivo .mp4
# # Ruta directa al ejecutable ffmpeg.exe

# #plt.rcParams['animation.ffmpeg_path'] = r"C:\Users\tooor\Downloads\ffmpeg-7.1.1-essentials_build\ffmpeg-7.1.1-essentials_build\bin\ffmpeg.exe"
# #writer = FFMpegWriter(fps=30, metadata=dict(artist='Tu Nombre'), bitrate=1800)
# #ani.save("trayectoria_hacia_atras.mp4", writer=writer)

# plt.show()

# ------------------------------------------------------------------------------
# Generación del database paralelizando
# ------------------------------------------------------------------------------
#Función base que se va a iterar constantemente
sim_cache_file = f"sim_base_cached_{integrator_name}_dt{dt_label}_hasta{end_year}.pkl"

# --------------------------------------------------------------------------
# FUNCIÓN PARA CREAR SIMULACIÓN BASE Y GUARDAR EN .PKL
# --------------------------------------------------------------------------
def crear_simulacion_base():
    print(f"⚙️ Generando nueva simulación base y guardando en caché: {sim_cache_file}")
    sim = rebound.Simulation()
    sim.units = ('AU', 'yr', 'Msun')
    sim.add(["Sun", "Mercury", "Venus", "Earth", "Mars",
             "Jupiter", "Saturn", "Uranus", "Neptune"], date=f"{end_year}-01-01")
    sim.integrator = integrator_name
    sim.dt = dt
    with open(sim_cache_file, "wb") as f:
        pickle.dump(sim, f)
    print("✅ Simulación base guardada.")

# --------------------------------------------------------------------------
# CARGAR SIMULACIÓN BASE DESDE .PKL
# --------------------------------------------------------------------------
def cargar_simulacion_base():
    if os.path.exists(sim_cache_file):
        try:
            with open(sim_cache_file, "rb") as f:
                sim = pickle.load(f)
            print(f"📦 Simulación base cargada desde caché: {sim_cache_file}")
            return sim
        except (EOFError, pickle.UnpicklingError):
            print("⚠️ Archivo de caché dañado. Regenerando...")
            os.remove(sim_cache_file)
            crear_simulacion_base()
            return cargar_simulacion_base()
    else:
        crear_simulacion_base()
        return cargar_simulacion_base()

# --------------------------------------------------------------------------
# FUNCIÓN PARA SIMULAR ASTEROIDE (usada en paralelo)
# --------------------------------------------------------------------------
def simular_asteroide(idx):
    with open(sim_cache_file, "rb") as f:
        sim = pickle.load(f)

    earth = sim.particles[3]

    v_kms = random.uniform(15, 45)
    v_auyr = v_kms / 4.74047
    phi = random.uniform(0, 2 * np.pi)
    theta = random.uniform(0, np.pi)
    vx = v_auyr * np.sin(theta) * np.cos(phi)
    vy = v_auyr * np.sin(theta) * np.sin(phi)
    vz = v_auyr * np.cos(theta)

    sim.add(x=earth.x, y=earth.y, z=earth.z,
            vx=earth.vx + vx, vy=earth.vy + vy, vz=earth.vz + vz,
            m=0.0)

    start_time = time.time()
    sim.integrate(sim.t + n_steps * dt)
    end_time = time.time()

    ast = sim.particles[-1]
    print(f"✅ Asteroide {idx} completado")
    return {
        "asteroide": idx,
        "x_final": ast.x,
        "y_final": ast.y,
        "z_final": ast.z,
        "vx_final": ast.vx,
        "vy_final": ast.vy,
        "vz_final": ast.vz,
        "tiempo_s": round(end_time - start_time, 4)
    }

# --------------------------------------------------------------------------
# EJECUCIÓN PRINCIPAL
# --------------------------------------------------------------------------
if __name__ == "__main__":
    # Asegurar que el caché esté generado y válido ANTES del multiprocessing
    _ = cargar_simulacion_base()

    with Pool(processes=10) as pool:
        resultados = pool.map(simular_asteroide, range(num_asteroides))

    df = pd.DataFrame(resultados)

    csv_name = f"resultados_{num_asteroides}ast_dt{dt_label}_hasta{end_year}.csv"
    npy_name = f"resultados_{num_asteroides}ast_dt{dt_label}_hasta{end_year}.npy"

    df.to_csv(csv_name, index=False)
    np.save(npy_name, df.to_records(index=False))

    print(f"\n✅ Guardado: {csv_name} y {npy_name}")

