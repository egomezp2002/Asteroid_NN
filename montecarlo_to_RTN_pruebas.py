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

muestra = df.iloc[144]   # fila número X


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

def simulacion_nominal(distancia_check_UA=1e-2, n_checks=118625):
    """
    Ejecuta la simulación SIN perturbaciones.
    Devuelve (impacto, x0, y0, z0, xf, yf, zf)
    """

    sim = cargar_simulacion_base().copy()
    sim.integrator = integrator_name
    sim.dt = dt

    # Condiciones nominales (sin ruido)
    x, y, z = muestra['x_final'], muestra['y_final'], muestra['z_final']
    vx, vy, vz = muestra['vx_final'], muestra['vy_final'], muestra['vz_final']

    x0, y0, z0 = x, y, z

    sim.add(x=x, y=y, z=z, vx=vx, vy=vy, vz=vz, m=0)

    earth = sim.particles[3]
    ast   = sim.particles[-1]

    tiempos = np.linspace(start_year, end_year, n_checks)
    thr2 = distancia_check_UA**2
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
    vxf, vyf, vzf = ast.vx, ast.vy, ast.vz

    return impacto, x0, y0, z0, xf, yf, zf, vxf, vyf, vzf

def montecarlo_paralelo(n_iter=1000, sigma_pos=1e-3, sigma_vel=1e-5, 
                        distancia_check_UA=1e-2, n_checks=118625):

    # --- Simulación nominal ---
    nominal = simulacion_nominal(distancia_check_UA, n_checks)
    impacto_nom, x0_nom, y0_nom, z0_nom, xf_nom, yf_nom, zf_nom, vxf_nom, vyf_nom, vzf_nom = nominal

    # --- Monte Carlo ---
    seeds = np.arange(n_iter, dtype=int)
    args = [(int(seed), sigma_pos, sigma_vel, distancia_check_UA, n_checks) for seed in seeds]

    sim_base = cargar_simulacion_base()
    sim_bytes = pickle.dumps(sim_base, protocol=pickle.HIGHEST_PROTOCOL)

    num_procs = max(1, cpu_count() - 2)
    with Pool(processes=num_procs, initializer=_pool_init, initargs=(sim_bytes,)) as pool:
        resultados = pool.map(simular_un_asteroide, args)

    impactos = [r[0] for r in resultados]
    pos_ini  = np.array([[r[1], r[2], r[3]] for r in resultados])
    pos_fin  = np.array([[r[4], r[5], r[6]] for r in resultados])

    prob = 100.0 * (sum(impactos) / float(n_iter))

    return {
        "probabilidad_mc": prob,
        "impactos_mc": impactos,
        "pos_ini_mc": pos_ini,
        "pos_fin_mc": pos_fin,
        "impacto_nominal": impacto_nom,
        "pos_ini_nominal": np.array([x0_nom, y0_nom, z0_nom]),
        "pos_fin_nominal": np.array([xf_nom, yf_nom, zf_nom]),
        "vel_nominal": np.array([vxf_nom, vyf_nom, vzf_nom])   # <-- Añadido
    }
# --------------------------------------------------------------------------
# USO
# --------------------------------------------------------------------------
if __name__ == "__main__":
    cargar_simulacion_base()

    resultados = montecarlo_paralelo(
        n_iter=3000, 
        sigma_pos=1e-5, 
        sigma_vel=1e-6,
        distancia_check_UA=1e-2,
        n_checks=118625
    )

    print(f"Probabilidad estimada de impacto (MC): {resultados['probabilidad_mc']:.2f}%")
    def get_rotation_matrix_C(r, v):
        """
        Devuelve la matriz de transformación de ECI -> RTN.
        r: vector posición nominal [x, y, z]
        v: vector velocidad nominal [vx, vy, vz]
        """
        r_unit = r / np.linalg.norm(r)
        h = np.cross(r, v)
        h_unit = h / np.linalg.norm(h)
        t_unit = np.cross(h_unit, r_unit)
        return np.column_stack((r_unit, t_unit, h_unit))
    
    # Nominal: posición y velocidad final
    r_nominal = resultados["pos_fin_nominal"]
    v_nominal = resultados["vel_nominal"]

    # Matriz de transformación
    C = get_rotation_matrix_C(r_nominal, v_nominal)

    # Transformar todos los puntos finales Monte Carlo
    dx = resultados["pos_fin_mc"] - r_nominal  # vector desde nominal
    pos_rtn = (C.T @ dx.T).T

    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    # --- Filtrar puntos fuera de 5σ ---
    sigma = np.std(pos_rtn, axis=0)        # σ por eje (Radial, In-track, Cross-track)
    mask = np.all(np.abs(pos_rtn) <= 6*sigma, axis=1)
    pos_rtn_filt = pos_rtn[mask]

    # --- 2D Covariance Ellipses ---
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    plot_pairs = [(1, 0), (1, 2), (0, 2)]  # (x, y) = In-track vs Radial, etc
    titles = ['Orbital Plane (In-track vs Radial)',
            'In-track vs Cross-track',
            'Radial vs Cross-track']
    x_labels = ['In-track [AU]', 'In-track [AU]', 'Radial [AU]']
    y_labels = ['Radial [AU]', 'Cross-track [AU]', 'Cross-track [AU]']
    colors = ['red', 'green', 'blue']

    for i, (idx_x, idx_y) in enumerate(plot_pairs):
        x = pos_rtn_filt[:, idx_x]
        y = pos_rtn_filt[:, idx_y]

        # Scatter Monte Carlo
        axs[i].scatter(x, y, s=10, c=colors[i], alpha=0.4, label='Monte Carlo Samples')

        # Covarianza y elipse
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

        axs[i].set_title(titles[i], fontsize=16, fontweight='bold')
        axs[i].set_xlabel(x_labels[i], fontsize=14, fontweight='bold')
        axs[i].set_ylabel(y_labels[i], fontsize=14, fontweight='bold')
        axs[i].grid(True, linestyle=':', alpha=0.6)
        if i == 0:
            axs[i].legend(fontsize=12)

    plt.suptitle("Cloud Dispersion in RTN Coordinates with Covariance Ellipses",
                fontsize=20, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # --- 3D Scatter ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    r = pos_rtn_filt[:, 0]
    t = pos_rtn_filt[:, 1]
    n = pos_rtn_filt[:, 2]

    ax.scatter(r, t, n, s=5, c='purple', alpha=0.4, label='Monte Carlo Samples')

    ax.set_xlabel('Radial [AU]', fontweight='bold')
    ax.set_ylabel('In-track [AU]', fontweight='bold')
    ax.set_zlabel('Cross-track [AU]', fontweight='bold')
    ax.set_title('3D Cloud Dispersion in RTN Coordinates', fontsize=16, fontweight='bold')

    # Escala uniforme en 3D
    max_range = np.array([r.max()-r.min(), t.max()-t.min(), n.max()-n.min()]).max() / 2.0
    mid_r = (r.max()+r.min()) * 0.5
    mid_t = (t.max()+t.min()) * 0.5
    mid_n = (n.max()+n.min()) * 0.5

    ax.set_xlim(mid_r - max_range, mid_r + max_range)
    ax.set_ylim(mid_t - max_range, mid_t + max_range)
    ax.set_zlim(mid_n - max_range, mid_n + max_range)

    ax.legend()
    plt.show()