import numpy as np
from datetime import datetime
from solar_simulation import simulate_solar_system_hourly
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
import rebound

# Nombres reales de los cuerpos (en el mismo orden que en la simulación)
planet_names = ["Sun", "Mercury", "Venus", "Earth", "Mars",
                "Jupiter", "Saturn", "Uranus", "Neptune"]

# Año final y año actual
end_year = 2028
start_year = datetime.now().year

# Ejecutar simulación
print(f"Simulando el sistema solar hora a hora desde {start_year} hasta {end_year}...")
data = simulate_solar_system_hourly(end_year=end_year)
# El tiempo se guarda en dias julianos relativos

# Construir nombres de archivos, guardar en memoria para simulaciones muy largas
filename_base = f"planet_positions_hourly_until_{end_year}"
npy_file = f"{filename_base}.npy"
csv_file = f"{filename_base}.csv"
# Guardar datos
np.save(npy_file, data)
np.savetxt(csv_file, data, delimiter=",")
print(f"✅ Datos guardados en:\n- {npy_file}\n- {csv_file}")

# -----------------------------------------------------------------------------------
# ANIMACIÓN (opcional). Descomenta esta sección si quieres visualizar las órbitas propagadas
# -----------------------------------------------------------------------------------
"""
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Calcular estructura de datos
n_cols = data.shape[1]
n_bodies = (n_cols - 1) // 3
planet_coords = np.array_split(data[:, 1:], n_bodies, axis=1)

# Crear figura
fig, ax = plt.subplots(figsize=(8, 8))
colors = plt.cm.viridis(np.linspace(0, 1, n_bodies))
lines, points = [], []

# Inicializar elementos gráficos
for i in range(n_bodies):
    line, = ax.plot([], [], lw=1, color=colors[i])
    point, = ax.plot([], [], 'o', color=colors[i], label=planet_names[i])
    lines.append(line)
    points.append(point)

# Etiquetas y diseño
ax.set_xlim(-35, 35)
ax.set_ylim(-35, 35)
ax.set_aspect('equal')
ax.set_title("Sistema Solar (proyección XY)")
ax.set_xlabel("X (UA)")
ax.set_ylabel("Y (UA)")
ax.legend(loc="upper right")

# Texto para el año en pantalla
year_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12)

# Función de actualización para animación
def update(frame):
    for i in range(n_bodies):
        x = planet_coords[i][:frame, 0]
        y = planet_coords[i][:frame, 1]
        lines[i].set_data(x, y)
        if len(x) > 0:
            points[i].set_data([x[-1]], [y[-1]])

    current_days = data[frame, 0]
    current_year = start_year + (current_days / 365.25)
    year_text.set_text(f"Año: {int(current_year)}")

    return lines + points + [year_text]

# Crear y mostrar animación
ani = FuncAnimation(fig, update, frames=range(100, len(data), 100), interval=1, blit=True)
plt.show()
"""
####################################
# Propagación hacia atrás del asteroide con Rebound

start_year = datetime.now().year
years_back = end_year - start_year

# Inicializar simulación
sim = rebound.Simulation()
sim.units = ('AU', 'yr', 'Msun')
sim.add(["Sun", "Mercury", "Venus", "Earth", "Mars",
         "Jupiter", "Saturn", "Uranus", "Neptune"], date=f"{end_year}-01-01")

# Obtener la Tierra (asumiendo que es el cuarto planeta tras el Sol)
earth = sim.particles[3]

# Generar velocidad de impacto aleatoria (15–45 km/s) convertida a AU/año
v_kms = random.uniform(15, 45)
v_auyr = v_kms / 4.74047

# Dirección aleatoria en el plano XY
angle = random.uniform(0, 2*np.pi)
vx = -v_auyr * np.cos(angle)
vy = -v_auyr * np.sin(angle)

# Añadir satélite en posición de la Tierra con velocidad relativa
sim.add(x=earth.x, y=earth.y, z=earth.z,
        vx=earth.vx + vx, vy=earth.vy + vy, vz=earth.vz,
        m=0.0)

# Configuración del integrador y propagación hacia atrás
sim.integrator = "whfast"
n_steps = 10000
dt = -years_back / n_steps  # paso negativo para ir hacia atrás

# Guardar posiciones en cada paso
positions = np.zeros((n_steps, sim.N, 2))  # solo plano XY

for i in range(n_steps):
    sim.integrate(sim.t + dt)
    for j, p in enumerate(sim.particles):
        positions[i, j, 0] = p.x
        positions[i, j, 1] = p.y

# Preparar animación con matplotlib
# Nombres reales de los cuerpos (incluye satélite como último)
planet_names = ["Sun", "Mercury", "Venus", "Earth", "Mars",
                "Jupiter", "Saturn", "Uranus", "Neptune", "Satellite"]

# Preparar animación con matplotlib
fig, ax = plt.subplots(figsize=(8, 8))
colors = ['yellow', 'gray', 'orange', 'blue', 'red', 'brown', 'gold', 'cyan', 'blueviolet', 'black']

# Crear líneas y puntos
lines = [ax.plot([], [], color=colors[i], lw=1)[0] for i in range(len(planet_names))]
points = [ax.plot([], [], 'o', color=colors[i], label=planet_names[i])[0] for i in range(len(planet_names))]

# Ajustes visuales
ax.set_xlim(-35, 35)
ax.set_ylim(-35, 35)
ax.set_aspect('equal')
ax.set_title(f"Propagación hacia atrás desde {end_year}")
ax.legend(loc='upper right')

# Texto del año en la parte superior
year_text = ax.text(0.5, 1.02, '', transform=ax.transAxes,
                    ha='center', va='bottom', fontsize=14, fontweight='bold')

# Función de actualización corregida
def update(frame):
    for i in range(sim.N):
        x = positions[:frame, i, 0]
        y = positions[:frame, i, 1]
        lines[i].set_data(x, y)
        points[i].set_data([positions[frame-1, i, 0]], [positions[frame-1, i, 1]])

    # Calcular el año actual
    current_fraction = frame / n_steps
    current_year = end_year - current_fraction * years_back
    year_text.set_text(f"Año: {current_year:.1f}")

    return lines + points + [year_text]

# Crear y mostrar animación
ani = FuncAnimation(fig, update, frames=range(10, n_steps, 10), interval=30, blit=False)

plt.show()
