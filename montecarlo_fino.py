import os
import pickle
import random
import time
from datetime import datetime

import numpy as np
import pandas as pd
import rebound
from multiprocessing import Pool, cpu_count

# ------------------------------------------------------------
# Configuración / constantes
# ------------------------------------------------------------
rebound.horizons.SSL_CONTEXT = "unverified"
BASE_SIM_BYTES = None  # caché en memoria para los workers

end_year = 2350
start_year = 2025
years_back = end_year - start_year
n_steps = 200000
dt = -years_back / n_steps            # tu valor original (negativo). Si quieres, usa abs(...)
num_asteroides = 20000
dt_label = f"{abs(dt):.6f}".replace('.', '_')
integrator_name = 'whfast'

planet_names = ["Sun", "Mercury", "Venus", "Earth", "Mars",
                "Jupiter", "Saturn", "Uranus", "Neptune", "Satellite"]

# ------------------------------------------------------------
# Cargar dataset y seleccionar muestra
# ------------------------------------------------------------
try:
    df = pd.read_csv("resultados_20000ast_dt0_001625_hasta2350.csv")
except Exception:
    df = pd.DataFrame(np.load("resultados_20000ast_dt0_001625_hasta2350.npy", allow_pickle=True))

muestra = df.iloc[388]   # fila a usar

# (opcional) extraer elementos orbitales si existen en el DF
try:
    a_base     = muestra['a']
    e_base     = muestra['e']
    inc_base   = muestra['i']
    Omega_base = muestra['Omega']
    omega_base = muestra['omega']
except KeyError:
    # No es obligatorio para el flujo Monte Carlo + ML
    pass

# ------------------------------------------------------------
# Simulación base cacheada
# ------------------------------------------------------------
sim_cache_file = "simulacion_base.pkl"

def crear_simulacion_base():
    print(f"⚙️ Generando nueva simulación base y guardando en caché: {sim_cache_file}")
    sim = rebound.Simulation()
    sim.units = ('AU', 'yr', 'Msun')
    sim.add(["Sun", "Mercury", "Venus", "Earth", "Mars",
             "Jupiter", "Saturn", "Uranus", "Neptune"], date=f"{start_year}-01-01")
    sim.integrator = integrator_name
    sim.dt = dt
    with open(sim_cache_file, "wb") as f:
        pickle.dump(sim, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("✅ Simulación base guardada.")

def cargar_simulacion_base():
    if os.path.exists(sim_cache_file):
        try:
            with open(sim_cache_file, "rb") as f:
                sim = pickle.load(f)
            return sim
        except (EOFError, pickle.UnpicklingError):
            print("⚠️ Caché dañada. Regenerando...")
            try:
                os.remove(sim_cache_file)
            except Exception:
                pass
            crear_simulacion_base()
            return cargar_simulacion_base()
    else:
        crear_simulacion_base()
        return cargar_simulacion_base()

def _pool_init(sim_bytes_in):
    """Inicializador para el Pool: deja la simulación base en RAM (bytes)."""
    global BASE_SIM_BYTES
    BASE_SIM_BYTES = sim_bytes_in

# ------------------------------------------------------------
# Monte Carlo (paralelo) — ahora devuelve posiciones y velocidades
# ------------------------------------------------------------
def simular_un_asteroide(args):
    """
    Ejecuta UNA simulación Monte Carlo con perturbaciones.
    Devuelve:
        impacto (0/1),
        x0,y0,z0, vx0,vy0,vz0,
        xf,yf,zf, vxf,vyf,vzf
    """
    seed, sigma_pos, sigma_vel, distancia_check_UA, n_checks = args
    np.random.seed(seed)

    # Recuperar simulación base
    global BASE_SIM_BYTES
    if BASE_SIM_BYTES is not None:
        sim = pickle.loads(BASE_SIM_BYTES)
    else:
        sim = cargar_simulacion_base()
    sim = sim.copy()
    sim.integrator = integrator_name
    sim.dt = dt

    # Estado base del asteroide (del registro 'muestra')
    x, y, z  = muestra['x_final'],  muestra['y_final'],  muestra['z_final']
    vx, vy, vz = muestra['vx_final'], muestra['vy_final'], muestra['vz_final']

    # Perturbaciones
    x  += np.random.normal(0, sigma_pos)
    y  += np.random.normal(0, sigma_pos)
    z  += np.random.normal(0, sigma_pos)
    vx += np.random.normal(0, sigma_vel)
    vy += np.random.normal(0, sigma_vel)
    vz += np.random.normal(0, sigma_vel)

    x0, y0, z0 = x, y, z
    vx0, vy0, vz0 = vx, vy, vz

    # Añadir asteroide como test particle
    sim.add(x=x, y=y, z=z, vx=vx, vy=vy, vz=vz, m=0)
    earth = sim.particles[3]
    ast   = sim.particles[-1]

    # Integración y chequeo de encuentro
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
    vxf, vyf, vzf = ast.vx, ast.vy, ast.vz

    return (impacto, x0, y0, z0, vx0, vy0, vz0, xf, yf, zf, vxf, vyf, vzf)

def montecarlo_paralelo(n_iter=1000, sigma_pos=1e-3, sigma_vel=1e-5,
                        distancia_check_UA=1e-2, n_checks=118625):
    """
    Devuelve:
      prob (%), pos_ini (N,3), vel_ini (N,3), pos_fin (N,3), vel_fin (N,3), impactos (N,)
    """
    seeds = np.arange(n_iter, dtype=int)
    args = [(int(seed), sigma_pos, sigma_vel, distancia_check_UA, n_checks) for seed in seeds]

    # Compartir simulación base como bytes al Pool
    sim_base = cargar_simulacion_base()
    sim_bytes = pickle.dumps(sim_base, protocol=pickle.HIGHEST_PROTOCOL)

    num_procs = max(1, cpu_count() - 2)  # deja 2 hilos libres
    with Pool(processes=num_procs, initializer=_pool_init, initargs=(sim_bytes,)) as pool:
        resultados = pool.map(simular_un_asteroide, args)

    resultados = np.array(resultados, dtype=float)

    impactos = resultados[:, 0].astype(int)
    pos_ini  = resultados[:, 1:4]
    vel_ini  = resultados[:, 4:7]
    pos_fin  = resultados[:, 7:10]
    vel_fin  = resultados[:, 10:13]

    prob = 100.0 * (impactos.sum() / float(n_iter))
    return prob, pos_ini, vel_ini, pos_fin, vel_fin, impactos

# ------------------------------------------------------------
# Conversión cartesiano -> elementos orbitales heliocéntricos
# ------------------------------------------------------------
def cartesian_to_orbital(sim, x, y, z, vx, vy, vz):
    """
    Convierte (x,y,z,vx,vy,vz) a elementos heliocéntricos (a, e, inc, Ω, ω)
    usando la API antigua: ast.orbit(primary=sun).
    - No mueve el tiempo de la simulación.
    - Añade y elimina temporalmente una partícula test.
    """
    sun = sim.particles[0]           # Sol
    idx_new = len(sim.particles)     # índice donde quedará el asteroide temporal
    sim.add(x=x, y=y, z=z, vx=vx, vy=vy, vz=vz, m=0)
    try:
        ast = sim.particles[-1]
        orb = ast.orbit(primary=sun)  # <-- API compatible con tu versión
        return orb.a, orb.e, orb.inc, orb.Omega, orb.omega
    finally:
        # Quita siempre la partícula temporal, incluso si hay error arriba
        sim.remove(index=idx_new)


def batch_cartesian_to_orbital(sim, pos, vel):
    """
    Convierte arrays (N,3) de posiciones y velocidades a (N,5) elementos:
    [a, e, inc, Ω, ω], con órbitas heliocéntricas.
    """
    N = pos.shape[0]
    out = np.empty((N, 5), dtype=float)
    for i in range(N):
        a, e, inc, Om, om = cartesian_to_orbital(
            sim,
            pos[i, 0], pos[i, 1], pos[i, 2],
            vel[i, 0], vel[i, 1], vel[i, 2]
        )
        out[i] = (a, e, inc, Om, om)
    return out


# ------------------------------------------------------------
# ML: carga del modelo (si existe) y predicción masiva
# ------------------------------------------------------------
def cargar_modelo_ml():
    """Carga modelo Keras desde .keras o .h5 si existe; devuelve None si no hay modelo."""
    try:
        import tensorflow as tf
        from tensorflow import keras
        load_model = keras.models.load_model
        print(f"[ML] TensorFlow {tf.__version__} importado correctamente.")
    except Exception as e:
        print(f"⚠️ TensorFlow no disponible: {e}")
        return None

    modelo_path = None
    for fname in ("modelo_entrenado.keras", "modelo_entrenado.h5"):
        if os.path.exists(fname):
            modelo_path = fname
            break
    if modelo_path is None:
        print("⚠️ No encontré 'modelo_entrenado.keras' ni 'modelo_entrenado.h5'. "
              "Se omite la predicción ML.")
        return None

    try:
        print(f"[ML] Cargando modelo: {modelo_path}")
        modelo = load_model(modelo_path)
        return modelo
    except Exception as e:
        if str(modelo_path).endswith(".h5") and ("h5py" in str(e).lower()):
            print("⚠️ Para cargar .h5 necesitas 'h5py'. Instala con: pip install h5py")
        else:
            print(f"⚠️ Error cargando el modelo: {e}")
        return None

def predecir_ml(modelo, X):
    """
    Predice con el modelo Keras:
      - devuelve (prob_media, tasa_clase_1, preds_raw)
      - asume salida sigmoide o similar en [0,1]
    """
    if modelo is None:
        return None, None, None
    y_pred = modelo.predict(X, verbose=0)
    y_pred = np.ravel(y_pred).astype(float)
    prob_media   = float(np.mean(y_pred))
    tasa_clase_1 = float(np.mean((y_pred >= 0.5).astype(int)))
    return prob_media, tasa_clase_1, y_pred

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    cargar_simulacion_base()

    # 1) Monte Carlo
    prob_mc, pos_ini, vel_ini, pos_fin, vel_fin, impactos = montecarlo_paralelo(
        n_iter=500,
        sigma_pos=1e-3,
        sigma_vel=1e-5,
        distancia_check_UA=1e-2,
        n_checks=118625
    )
    print(f"Probabilidad estimada de impacto (Monte Carlo): {prob_mc:.2f}%")

    # 2) Convertir estados FINALES a elementos heliocéntricos
    sim_ref = cargar_simulacion_base()
    elems_fin = batch_cartesian_to_orbital(sim_ref, pos_fin, vel_fin)  # (N,5) -> a,e,inc,Ω,ω
    a_vals, e_vals, inc_vals, Om_vals, om_vals = elems_fin.T

    # 3) Cargar modelo y evaluar TODOS los puntos del Monte Carlo
    modelo = cargar_modelo_ml()
    if modelo is not None:

        X_ml = np.column_stack([a_vals, e_vals, inc_vals, Om_vals, om_vals])  # (N,5)

        prob_media_ml, tasa_cls1_ml, _ = predecir_ml(modelo, X_ml)
        if prob_media_ml is not None:
            print(f"🔮 Modelo ML → probabilidad media (sobre {len(X_ml)} puntos): "
                  f"{100*prob_media_ml:.2f}%")
            print(f"🔮 Modelo ML → tasa clasificada como impacto (umbral 0.5): "
                  f"{100*tasa_cls1_ml:.2f}%")
    else:
        print("ℹ️ Sin modelo ML cargado; se muestra solo el resultado de Monte Carlo.")
