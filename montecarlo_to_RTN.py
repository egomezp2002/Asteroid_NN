

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
from multiprocessing import cpu_count

rebound.horizons.SSL_CONTEXT = "unverified"  
BASE_SIM_BYTES = None  # caché en memoria para los workers


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

muestra = df.iloc[182]   # fila número X


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


def _pool_init(sim_bytes_in):
    global BASE_SIM_BYTES
    BASE_SIM_BYTES = sim_bytes_in


# --------------------------------------------------------------------------
# MONTECARLO CON SIMULACIÓN BASE CACHEADA
# --------------------------------------------------------------------------
from multiprocessing import Pool, cpu_count

def simular_un_asteroide(args):
    """
    Ejecuta UNA simulación Monte Carlo con perturbaciones.
    Devuelve (impacto, x0, y0, z0, xf, yf, zf).
    """
    seed, sigma_pos, sigma_vel, distancia_check_UA, n_checks = args
    np.random.seed(seed)
    if seed == 0:
        print("DEBUG sigma_pos:", sigma_pos, "sigma_vel:", sigma_vel)

    global BASE_SIM_BYTES
    if BASE_SIM_BYTES is not None:
        sim = pickle.loads(BASE_SIM_BYTES)
    else:
        sim = cargar_simulacion_base()
    sim = sim.copy()

    sim.integrator = integrator_name
    sim.dt = dt

    # Posición/velocidad inicial base
    x, y, z = muestra['x_final'], muestra['y_final'], muestra['z_final']
    vx, vy, vz = muestra['vx_final'], muestra['vy_final'], muestra['vz_final']

    # Perturbaciones
    x  += np.random.normal(0, sigma_pos)
    y  += np.random.normal(0, sigma_pos)
    z  += np.random.normal(0, sigma_pos)
    vx += np.random.normal(0, sigma_vel)
    vy += np.random.normal(0, sigma_vel)
    vz += np.random.normal(0, sigma_vel)

    x0, y0, z0 = x, y, z  # guardar posición inicial perturbada

    sim.add(x=x, y=y, z=z, vx=vx, vy=vy, vz=vz, m=0)
    earth = sim.particles[3]
    ast   = sim.particles[-1]

    tiempos = np.linspace(start_year, end_year, n_checks)
    thr2 = (distancia_check_UA)**2
    impacto = 0

    for t in tiempos:
        sim.integrate(t, exact_finish_time=False)
        dx = earth.x - ast.x
        dy = earth.y - ast.y
        dz = earth.z - ast.z
        if (dx*dx + dy*dy + dz*dz) < thr2:
            impacto = 1
            break

    xf, yf, zf = ast.x, ast.y, ast.z
    return impacto, x0, y0, z0, xf, yf, zf



def montecarlo_paralelo(n_iter=1000, sigma_pos=1e-3, sigma_vel=1e-5, 
                        distancia_check_UA=1e-2, n_checks=118625):
    seeds = np.arange(n_iter, dtype=int)
    args = [(int(seed), sigma_pos, sigma_vel, distancia_check_UA, n_checks) for seed in seeds]

    sim_base = cargar_simulacion_base()
    sim_bytes = pickle.dumps(sim_base, protocol=pickle.HIGHEST_PROTOCOL)

    num_procs = max(1, cpu_count() - 2)
    with Pool(processes=num_procs, initializer=_pool_init, initargs=(sim_bytes,)) as pool:
        resultados = pool.map(simular_un_asteroide, args)

    # Desempaquetar resultados
    impactos = [r[0] for r in resultados]
    pos_ini  = np.array([[r[1], r[2], r[3]] for r in resultados])
    pos_fin  = np.array([[r[4], r[5], r[6]] for r in resultados])

    prob = 100.0 * (sum(impactos) / float(n_iter))
    return prob, pos_ini, pos_fin


# --------------------------------------------------------------------------
# USO
# --------------------------------------------------------------------------
if __name__ == "__main__":
    cargar_simulacion_base()

    prob, pos_ini, pos_fin = montecarlo_paralelo(
        n_iter=1200, 
        sigma_pos=1e-5, 
        sigma_vel=1e-6,
        distancia_check_UA=1e-2,
        n_checks=118625
    )
    print(f"Probabilidad estimada de impacto: {prob:.2f}%")

    def set_equal_3d_from_points(ax, X, Y, Z, pad_ratio=0.05):
        xmin, xmax = np.min(X), np.max(X)
        ymin, ymax = np.min(Y), np.max(Y)
        zmin, zmax = np.min(Z), np.max(Z)
        cx = 0.5*(xmin+xmax); cy = 0.5*(ymin+ymax); cz = 0.5*(zmin+zmax)
        ranges = np.array([xmax-xmin, ymax-ymin, zmax-zmin], dtype=float)
        R = np.max(ranges)
        if R == 0: R = 1e-9
        R *= (1 + pad_ratio)
        ax.set_xlim(cx - R/2, cx + R/2)
        ax.set_ylim(cy - R/2, cy + R/2)
        ax.set_zlim(cz - R/2, cz + R/2)
        try:
            ax.set_box_aspect([1,1,1])
        except Exception:
            pass

    def bbox_from_points(P, scale=1.05):
        X, Y, Z = P[:,0], P[:,1], P[:,2]
        xmin, xmax = X.min(), X.max()
        ymin, ymax = Y.min(), Y.max()
        zmin, zmax = Z.min(), Z.max()
        cx = 0.5*(xmin+xmax); cy = 0.5*(ymin+ymax); cz = 0.5*(zmin+zmax)
        R = max(xmax-xmin, ymax-ymin, zmax-zmin)
        if R == 0: R = 1e-9
        return (cx, cy, cz), 0.5*R*scale

    # --- Figura con dos paneles: global y zoom a iniciales ---
    fig = plt.figure(figsize=(12, 6))
    ax_global = fig.add_subplot(1, 2, 1, projection='3d')
    ax_zoom   = fig.add_subplot(1, 2, 2, projection='3d')

# Global (ambos conjuntos)
    ax_global.scatter(pos_ini[:,0], pos_ini[:,1], pos_ini[:,2],
              s=28, facecolors='none', edgecolors='blue',
              linewidths=0.9, label='Initial points', zorder=3)
    ax_global.scatter(pos_fin[:,0], pos_fin[:,1], pos_fin[:,2],
              s=12, c='red', alpha=0.7, label='Final points', zorder=2)

    X_all = np.concatenate([pos_ini[:,0], pos_fin[:,0]])
    Y_all = np.concatenate([pos_ini[:,1], pos_fin[:,1]])
    Z_all = np.concatenate([pos_ini[:,2], pos_fin[:,2]])
    set_equal_3d_from_points(ax_global, X_all, Y_all, Z_all, pad_ratio=0.05)

    ax_global.set_title("Final positions", fontsize=16, fontweight='bold')
    ax_global.set_xlabel("X [AU]", fontsize=16, fontweight='bold')
    ax_global.set_ylabel("Y [AU]", fontsize=16, fontweight='bold')
    ax_global.set_zlabel("Z [AU]", fontsize=16, fontweight='bold')
    ax_global.legend(loc='upper left')

# Zoom (solo iniciales)
    ax_zoom.scatter(pos_ini[:,0], pos_ini[:,1], pos_ini[:,2],
            s=40, facecolors='none', edgecolors='blue',
            linewidths=1.2, label='Iniciales', zorder=3)

    (cx, cy, cz), R = bbox_from_points(pos_ini, scale=1.10)
    ax_zoom.set_xlim(cx - R, cx + R)
    ax_zoom.set_ylim(cy - R, cy + R)
    ax_zoom.set_zlim(cz - R, cz + R)
    try:
        ax_zoom.set_box_aspect([1,1,1])
    except Exception:
        pass

    ax_zoom.set_title("Initial positions", fontsize=16, fontweight='bold')
    ax_zoom.set_xlabel("X [AU]", fontsize=16, fontweight='bold')
    ax_zoom.set_ylabel("Y [AU]", fontsize=16, fontweight='bold')
    ax_zoom.set_zlabel("Z [AU]", fontsize=16, fontweight='bold')
    ax_zoom.legend(loc='upper left')

    plt.suptitle("Monte Carlo: initial positions (blue) and final positions (red)",
             fontsize=20, fontweight='bold')
    plt.tight_layout()
    plt.show()

# --- Transformación a Coordenadas RTN ---
    import matplotlib.patches as patches

# 1. Estado nominal
# --- Datos nominales y transformación a RTN ---
    r_nominal = np.mean(pos_fin, axis=0)
    v_nominal = np.array([muestra['vx_final'], muestra['vy_final'], muestra['vz_final']])

    def get_rotation_matrix_C(r, v):
        r_unit = r / np.linalg.norm(r)
        h = np.cross(r, v)
        h_unit = h / np.linalg.norm(h)
        t_unit = np.cross(h_unit, r_unit)
        return np.column_stack((r_unit, t_unit, h_unit))

    C = get_rotation_matrix_C(r_nominal, v_nominal)
    dx = pos_fin - r_nominal
    pos_rtn = (C.T @ dx.T).T

# --- Plotting with Covariance Ellipses ---
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    plot_pairs = [(1, 0), (1, 2), (0, 2)]
    titles = ['Orbital Plane (In-track vs Radial)',
          'In-track vs Cross-track',
          'Radial vs Cross-track']
    x_labels = ['In-track [AU]', 'In-track [AU]', 'Radial [AU]']
    y_labels = ['Radial [AU]', 'Cross-track [AU]', 'Cross-track [AU]']
    colors = ['red', 'green', 'blue']

    for i, (idx_x, idx_y) in enumerate(plot_pairs):
        x = pos_rtn[:, idx_x]
        y = pos_rtn[:, idx_y]

        # --- Scatter de Monte Carlo ---
        axs[i].scatter(x, y, s=10, c=colors[i], alpha=0.4, label='Monte Carlo Samples')

        # --- Covarianza y elipse ---
        cov = np.cov(x, y)
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

        for n_sigma, ltype, alpha in zip([1, 3], ['--', '-'], [0.8, 0.3]):
            width, height = 2 * n_sigma * np.sqrt(vals)
            ell = patches.Ellipse(xy=(np.mean(x), np.mean(y)),
                              width=width, height=height, angle=theta,
                              edgecolor='black', lw=1.5, facecolor='none',
                              ls=ltype, alpha=alpha, label=f'{n_sigma}$\\sigma$ Bound')
            axs[i].add_patch(ell)

    # --- Etiquetas y títulos ---
            axs[i].set_title(titles[i], fontsize=16, fontweight='bold')
            axs[i].set_xlabel(x_labels[i], fontsize=14, fontweight='bold')
            axs[i].set_ylabel(y_labels[i], fontsize=14, fontweight='bold')
            axs[i].grid(True, linestyle=':', alpha=0.6)

    # Solo añadir leyenda en el primer subplot
        if i == 0:
            axs[i].legend(fontsize=12)

    plt.suptitle("Cloud Dispersion in RTN Coordinates with Covariance Ellipses",
             fontsize=20, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # --- 3D RTN Scatter Plot ---
    fig = plt.figure(figsize=(10, 8))
    # Le decimos a Matplotlib que esta será una gráfica 3D
    ax = fig.add_subplot(111, projection='3d')

    # Según tu código anterior: 0=Radial, 1=In-track, 2=Cross-track
    r = pos_rtn[:, 0]
    t = pos_rtn[:, 1]
    n = pos_rtn[:, 2]

    # Graficamos la nube de puntos
    ax.scatter(r, t, n, s=5, c='purple', alpha=0.4, label='Monte Carlo Samples')

    # Configuramos las etiquetas de los ejes
    ax.set_xlabel('Radial [AU]', fontweight='bold')
    ax.set_ylabel('In-track [AU]', fontweight='bold')
    ax.set_zlabel('Cross-track [AU]', fontweight='bold')
    ax.set_title('3D Cloud Dispersion in RTN Coordinates', fontsize=16, fontweight='bold')

    # --- TRUCO: Igualar la escala de los ejes ---
    # En 3D, si los ejes no tienen la misma escala, la nube se verá deformada artificialmente.
    # Esto fuerza a que la "caja" 3D sea un cubo perfecto.
    max_range = np.array([r.max()-r.min(), t.max()-t.min(), n.max()-n.min()]).max() / 2.0
    mid_r = (r.max()+r.min()) * 0.5
    mid_t = (t.max()+t.min()) * 0.5
    mid_n = (n.max()+n.min()) * 0.5

    ax.set_xlim(mid_r - max_range, mid_r + max_range)
    ax.set_ylim(mid_t - max_range, mid_t + max_range)
    ax.set_zlim(mid_n - max_range, mid_n + max_range)

    ax.legend()
    plt.show()