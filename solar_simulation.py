import rebound
from datetime import datetime
import numpy as np

def simulate_solar_system_hourly(end_year: int):
    """
    Simula el sistema solar desde hoy hasta el año indicado,
    guardando las posiciones de todos los planetas cada hora.

    Args:
        end_year (int): Año final de la simulación.

    Returns:
        np.ndarray: Array de forma (n_instantes, n_cuerpos * 3 + 1) con [tiempo, x1, y1, z1, ..., xN, yN, zN].
    """
    # Crear simulación e inicializar parámetros
    sim = rebound.Simulation()
    sim.units = ('AU', 'days', 'Msun')
    sim.integrator = "ias15"  # Precisión alta

    rebound.horizons.SSL_CONTEXT = 'unverified'

    # Añadir el Sol y los 8 planetas principales
    sim.add("Sun")
    for planet in ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]:
        sim.add(planet)

    sim.move_to_com()  # Centrar en el centro de masa

    # Calcular tiempo total de simulación
    start_year = datetime.now().year
    years = end_year - start_year
    total_days = int(years * 365.25)

    # Parámetros de integración
    timestep_days = 1.0 / 24.0  # 1 hora en días
    n_steps = int(total_days / timestep_days)

    # Inicializar lista para posiciones
    n_bodies = len(sim.particles)
    data = []

    # Integración con almacenamiento de posiciones
    for step in range(n_steps):
        current_time = sim.t  # Tiempo actual en días

        row = [current_time]  # Comenzamos con el tiempo
        for p in sim.particles:
            row.extend([p.x, p.y, p.z])  # Posición en 3D de cada cuerpo

        data.append(row)
        sim.integrate(sim.t + timestep_days)

    # Convertir a array NumPy
    data_array = np.array(data)

    return data_array
