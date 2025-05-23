
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

# Convierte velocidad de km/s a unidades astronómicas por día (AU/día)
def kmps_to_au_per_day(v_kmps):
    return v_kmps * 86400 / 149597870.7

# Genera un vector unitario aleatorio en 3D, uniformemente distribuido en una esfera
def random_unit_vector_3d():
    phi = np.random.uniform(0, 2*np.pi)
    cos_theta = np.random.uniform(-1, 1)
    sin_theta = np.sqrt(1 - cos_theta**2)
    return np.array([sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta])

# Propaga hacia atrás la trayectoria de un asteroide que impacta con la Tierra en 'end_year'
def propagate_asteroid_backward(end_year, filename, v_range_kmps=(15, 45)):
    """
    Simula hacia atrás la trayectoria de un asteroide que impacta con la Tierra en end_year.
    Usa posiciones reales de todos los planetas para calcular fuerzas gravitatorias.
    
    Args:
        end_year (int): Año del impacto simulado.
        filename (str): Archivo .npy con las posiciones del sistema solar hora a hora.
        v_range_kmps (tuple): Rango de velocidad inicial en km/s (mínimo, máximo).
    
    Returns:
        current_state (np.ndarray): [x, y, z, vx, vy, vz] del asteroide en la fecha actual.
        sol (OdeResult): Objeto de scipy con toda la trayectoria integrada hacia atrás.
    """

    # Cargar los datos simulados previamente (t, x1, y1, z1, ..., xN, yN, zN)
    data = np.load(filename)  # shape: (timesteps, 1 + 9*3)
    times = data[:, 0]        # Tiempo en días desde hoy hasta end_year

    n_bodies = (data.shape[1] - 1) // 3
    body_names = ["Sun", "Mercury", "Venus", "Earth", "Mars",
                  "Jupiter", "Saturn", "Uranus", "Neptune"]
    assert len(body_names) == n_bodies

    # Crear interpoladores cúbicos para (x, y, z) de cada cuerpo a lo largo del tiempo
    interpolators = {}
    for i, name in enumerate(body_names):
        x = data[:, 1 + i*3]
        y = data[:, 2 + i*3]
        z = data[:, 3 + i*3]
        interpolators[name] = {
            'x': interp1d(times, x, kind='cubic'),
            'y': interp1d(times, y, kind='cubic'),
            'z': interp1d(times, z, kind='cubic')
        }

    # Posición de la Tierra en el momento del impacto (última fila)
    earth_xyz = np.array([
        interpolators["Earth"]['x'](times[-1]),
        interpolators["Earth"]['y'](times[-1]),
        interpolators["Earth"]['z'](times[-1])
    ])

    # Generar dirección aleatoria y velocidad negativa (hacia atrás)
    direction = random_unit_vector_3d()
    speed_kmps = np.random.uniform(*v_range_kmps)
    speed = kmps_to_au_per_day(speed_kmps)
    v_xyz = -direction * speed  # movimiento hacia atrás en el tiempo

    # Masas de los planetas en masas solares (valores típicos)
    masses = {
        "Sun": 1.0,
        "Mercury": 1.65e-7,
        "Venus": 2.45e-6,
        "Earth": 3.00e-6,
        "Mars": 3.2e-7,
        "Jupiter": 0.000954,
        "Saturn": 0.000285,
        "Uranus": 4.37e-5,
        "Neptune": 5.15e-5
    }

    # Ecuaciones del movimiento del asteroide influenciado por todos los cuerpos
    def asteroid_dynamics(t, state):
        x, y, z, vx, vy, vz = state
        ax = ay = az = 0.0
        for name in body_names:
            pos = np.array([
                interpolators[name]['x'](t),
                interpolators[name]['y'](t),
                interpolators[name]['z'](t)
            ])
            diff = pos - np.array([x, y, z])
            r = np.linalg.norm(diff)
            if r > 1e-6:
                factor = masses[name] / r**3
                ax += factor * diff[0]
                ay += factor * diff[1]
                az += factor * diff[2]
        return [vx, vy, vz, ax, ay, az]

    # Estado inicial del asteroide en el momento del impacto
    state0 = np.concatenate([earth_xyz, v_xyz])

    # Definir intervalo de integración hacia atrás (desde impacto hasta hoy)
    t_span = [times[-1], times[0]]  # desde el final hacia el principio
    t_eval = np.linspace(*t_span, len(times))  # pasos temporales para guardar datos

    # Ejecutar integración numérica usando solve_ivp (RK45 adaptativo)
    sol = solve_ivp(
        asteroid_dynamics,
        t_span,
        state0,
        method='RK45',
        t_eval=t_eval,
        rtol=1e-9  # alta precisión relativa
    )

    # Devolver estado del asteroide en la fecha actual (t=0)
    current_state = sol.y[:, -1]
    return current_state, sol
