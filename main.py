import numpy as np
from datetime import datetime
from solar_simulation import simulate_solar_system_hourly
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# Nombres reales de los cuerpos (en el mismo orden que en la simulación)
planet_names = ["Sun", "Mercury", "Venus", "Earth", "Mars",
                "Jupiter", "Saturn", "Uranus", "Neptune"]

# Año final y año actual
end_year = 2026
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


# -----------------------------------------------------------------------------------
# Backwards propagation del asteroide que impacta
# -----------------------------------------------------------------------------------

from backwards_propagator import propagate_asteroid_backward

filename = f"planet_positions_hourly_until_{end_year}.npy"

# Cargar la trayectoria del asteroide
current_state, sol, launch_velocity = propagate_asteroid_backward(end_year, filename)
asteroid_xy = sol.y[0:2].T  # posiciones x, y en cada tiempo

# Dibujar trayectoria
plt.figure(figsize=(8, 8))
plt.plot(asteroid_xy[:, 0], asteroid_xy[:, 1], 'r-', label="Asteroide")
plt.plot(asteroid_xy[0, 0], asteroid_xy[0, 1], 'go', label="Inicio (impacto)")
plt.plot(asteroid_xy[-1, 0], asteroid_xy[-1, 1], 'bo', label="Estado actual")

plt.xlabel("X (UA)")
plt.ylabel("Y (UA)")
plt.title("Trayectoria del asteroide propagada hacia atrás")
plt.axis('equal')
plt.xlim(-35, 35)
plt.ylim(-35, 35)
plt.legend()
plt.grid(True)

# Añadir círculos para representar órbitas planetarias típicas
orbital_radii = [0.39, 0.72, 1.00, 1.52, 5.20, 9.58, 19.2, 30.1]
for r in orbital_radii:
    circle = plt.Circle((0, 0), r, color='gray', linestyle='--', linewidth=0.5, fill=False)
    plt.gca().add_patch(circle)

# Dibujar vector de velocidad en el punto de impacto
impact_x, impact_y = asteroid_xy[0]  # posición de impacto (primer punto)
vx, vy = launch_velocity[:2]         # componentes x, y de la velocidad

scale = 5  # ajusta para que la flecha se vea clara
plt.quiver(
    impact_x, impact_y, vx, vy,
    angles='xy', scale_units='xy', scale=1/scale,
    color='green', label='Vector de lanzamiento'
)

plt.show()