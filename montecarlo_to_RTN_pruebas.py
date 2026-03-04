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

muestra = df.iloc[273]   # fila número X


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
        return np.array([r_unit, t_unit, h_unit])
    
    # =========================================================================
    # ESTADO INICIAL: Cloud Dispersion in RTN Coordinates
    # =========================================================================
    import matplotlib.patches as patches

    
    # 1. Nominal inicial: posición y velocidad (extraídos directamente de la muestra)
    r_nominal_ini = np.array([muestra['x_final'], muestra['y_final'], muestra['z_final']])
    v_nominal_ini = np.array([muestra['vx_final'], muestra['vy_final'], muestra['vz_final']])

    # 2. Matriz de transformación inicial
    C_ini = get_rotation_matrix_C(r_nominal_ini, v_nominal_ini)

    # 3. Transformar todos los puntos iniciales Monte Carlo
    dx_ini = resultados["pos_ini_mc"] - r_nominal_ini  # vector desde nominal inicial
    pos_rtn_ini = dx_ini @ C_ini.T

    # --- Filtrar puntos fuera de 6σ ---
    sigma_ini = np.std(pos_rtn_ini, axis=0)        # σ por eje (Radial, In-track, Cross-track)
    mask_ini = np.all(np.abs(pos_rtn_ini) <= 6*sigma_ini, axis=1)
    pos_rtn_filt_ini = pos_rtn_ini[mask_ini]

    # --- 2D Covariance Ellipses ---
    fig_ini, axs_ini = plt.subplots(1, 3, figsize=(18, 6))

    plot_pairs = [(1, 0), (1, 2), (0, 2)]  # (x, y) = In-track vs Radial, etc
    titles_ini = ['Initial Orbital Plane (In-track vs Radial)',
                  'Initial In-track vs Cross-track',
                  'Initial Radial vs Cross-track']
    x_labels = ['In-track [AU]', 'In-track [AU]', 'Radial [AU]']
    y_labels = ['Radial [AU]', 'Cross-track [AU]', 'Cross-track [AU]']
    colors = ['red', 'green', 'blue']

    for i, (idx_x, idx_y) in enumerate(plot_pairs):
        x = pos_rtn_filt_ini[:, idx_x]
        y = pos_rtn_filt_ini[:, idx_y]

        # Scatter Monte Carlo
        axs_ini[i].scatter(x, y, s=10, c=colors[i], alpha=0.4, label='Monte Carlo Samples (Ini)')

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
            axs_ini[i].add_patch(ell)

        axs_ini[i].set_title(titles_ini[i], fontsize=16, fontweight='bold')
        axs_ini[i].set_xlabel(x_labels[i], fontsize=14, fontweight='bold')
        axs_ini[i].set_ylabel(y_labels[i], fontsize=14, fontweight='bold')
        axs_ini[i].grid(True, linestyle=':', alpha=0.6)
        axs_ini[i].set_aspect('equal', adjustable='box')
        if i == 0:
            axs_ini[i].legend(fontsize=12)

    plt.suptitle("Initial Cloud Dispersion in RTN Coordinates with Covariance Ellipses",
                 fontsize=20, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # --- 3D Scatter ---
    fig_ini_3d = plt.figure(figsize=(10, 8))
    ax_ini = fig_ini_3d.add_subplot(111, projection='3d')

    r_ini = pos_rtn_filt_ini[:, 0]
    t_ini = pos_rtn_filt_ini[:, 1]
    n_ini = pos_rtn_filt_ini[:, 2]

    ax_ini.scatter(r_ini, t_ini, n_ini, s=5, c='purple', alpha=0.4, label='Monte Carlo Samples (Ini)')

    # --- Esfera límite sigma_pos centrada en (0,0,0) ---
    sigma_pos = 1e-5

    # Parametrización esférica
    u = np.linspace(0, 2*np.pi, 60)
    v = np.linspace(0, np.pi, 60)

    x_sphere = sigma_pos * np.outer(np.cos(u), np.sin(v))
    y_sphere = sigma_pos * np.outer(np.sin(u), np.sin(v))
    z_sphere = sigma_pos * np.outer(np.ones(np.size(u)), np.cos(v))

    # Dibujar superficie
    ax_ini.plot_surface(x_sphere, y_sphere, z_sphere,
                        color='cyan', alpha=0.15, edgecolor='none')

    # Opcional: contorno más visible
    ax_ini.plot_wireframe(x_sphere, y_sphere, z_sphere,
                        color='black', linewidth=0.3, alpha=0.3)

    # Marcar el centro
    ax_ini.scatter(0, 0, 0, color='black', s=50, label=r'Centro (0,0,0)')

    ax_ini.set_xlabel('Radial [AU]', fontweight='bold')
    ax_ini.set_ylabel('In-track [AU]', fontweight='bold')
    ax_ini.set_zlabel('Cross-track [AU]', fontweight='bold')
    ax_ini.set_title('3D Initial Cloud Dispersion in RTN Coordinates', fontsize=16, fontweight='bold')

    # Escala uniforme en 3D
    max_range_ini = np.array([r_ini.max()-r_ini.min(), t_ini.max()-t_ini.min(), n_ini.max()-n_ini.min()]).max() / 2.0
    mid_r_ini = (r_ini.max()+r_ini.min()) * 0.5
    mid_t_ini = (t_ini.max()+t_ini.min()) * 0.5
    mid_n_ini = (n_ini.max()+n_ini.min()) * 0.5

    ax_ini.set_xlim(mid_r_ini - max_range_ini, mid_r_ini + max_range_ini)
    ax_ini.set_ylim(mid_t_ini - max_range_ini, mid_t_ini + max_range_ini)
    ax_ini.set_zlim(mid_n_ini - max_range_ini, mid_n_ini + max_range_ini)

    ax_ini.legend()
    ax_ini.set_box_aspect([1,1,1])  # Esto fuerza a que el cubo visual sea un cubo real
    plt.show()
    # RTN FINAL

    # Nominal: posición y velocidad final
    r_nominal = resultados["pos_fin_nominal"]
    v_nominal = resultados["vel_nominal"]

    # Matriz de transformación
    C = get_rotation_matrix_C(r_nominal, v_nominal)

    # Transformar todos los puntos finales Monte Carlo
    dx = resultados["pos_fin_mc"] - r_nominal  # vector desde nominal
    pos_rtn = dx @ C.T

    import numpy as np
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

    # --- 3D Scatter con esfera ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    r = pos_rtn_filt[:, 0]
    t = pos_rtn_filt[:, 1]
    n = pos_rtn_filt[:, 2]

    # Calcular distancia desde el origen
    dist = np.sqrt(r**2 + t**2 + n**2)
    radius = 1e-2  # 0.01 AU

    # Máscara de puntos dentro de la esfera
    mask_inside = dist <= radius

    # Scatter de puntos fuera de la esfera
    ax.scatter(r[~mask_inside], t[~mask_inside], n[~mask_inside], s=5, c='purple', alpha=0.4, label='Monte Carlo Samples')

    # Scatter de puntos dentro de la esfera
    ax.scatter(r[mask_inside], t[mask_inside], n[mask_inside], s=20, c='orange', alpha=0.8, label='Inside Sphere')

    # --- Dibujar la esfera ---
    u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:25j]
    x_sphere = radius * np.cos(u) * np.sin(v)
    y_sphere = radius * np.sin(u) * np.sin(v)
    z_sphere = radius * np.cos(v)
    ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color="blue", alpha=0.2, linewidth=1)

    ax.set_xlabel('Radial [AU]', fontweight='bold')
    ax.set_ylabel('In-track [AU]', fontweight='bold')
    ax.set_zlabel('Cross-track [AU]', fontweight='bold')
    ax.set_title('3D Cloud Dispersion in RTN Coordinates with Sphere', fontsize=16, fontweight='bold')

    # Escala uniforme
    max_range = np.array([r.max()-r.min(), t.max()-t.min(), n.max()-n.min()]).max() / 2.0
    mid_r = (r.max()+r.min()) * 0.5
    mid_t = (t.max()+t.min()) * 0.5
    mid_n = (n.max()+n.min()) * 0.5
    ax.set_xlim(mid_r - max_range, mid_r + max_range)
    ax.set_ylim(mid_t - max_range, mid_t + max_range)
    ax.set_zlim(mid_n - max_range, mid_n + max_range)

    ax.legend()
    ax.set_box_aspect([1,1,1])
    plt.show()