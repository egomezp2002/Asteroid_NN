
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
import rebound
import pandas as pd
from solar_simulation import simulate_solar_system_hourly
rebound.horizons.SSL_CONTEXT = "unverified"  

# ------------------------------------------------------------------------------
# CONFIGURACIÓN GLOBAL
# ------------------------------------------------------------------------------
end_year = 2026
start_year = datetime.now().year
years_back = end_year - start_year
n_steps = 100
dt = -years_back / n_steps
Gamma = 50 #Nro de asteroides del dataset

planet_names = ["Sun", "Mercury", "Venus", "Earth", "Mars",
                "Jupiter", "Saturn", "Uranus", "Neptune", "Satellite"]

# ------------------------------------------------------------------------------
# SIMULACIÓN DEL SISTEMA SOLAR (opcional) - Simular hacia delante y guardar
# ------------------------------------------------------------------------------
"""
print(f"Simulando el sistema solar hora a hora desde {start_year} hasta {end_year}...")
data = simulate_solar_system_hourly(end_year=end_year)
filename_base = f"planet_positions_hourly_until_{end_year}"
np.save(f"{filename_base}.npy", data)
np.savetxt(f"{filename_base}.csv", data, delimiter=",")
print(f"✅ Datos guardados en: {filename_base}.npy y .csv")
"""

# ------------------------------------------------------------------------------
# PROPAGACIÓN HACIA ATRÁS DE UN ASTEROIDE CON WHFAST
# ------------------------------------------------------------------------------
sim = rebound.Simulation()
sim.units = ('AU', 'yr', 'Msun')
sim.add(["Sun", "Mercury", "Venus", "Earth", "Mars",
         "Jupiter", "Saturn", "Uranus", "Neptune"], date=f"{end_year}-01-01")

earth = sim.particles[3]
v_kms = random.uniform(15, 45)
v_auyr = v_kms / 4.74047
angle = random.uniform(0, 2*np.pi)
vx = -v_auyr * np.cos(angle)
vy = -v_auyr * np.sin(angle)

sim.add(x=earth.x, y=earth.y, z=earth.z,
        vx=earth.vx + vx, vy=earth.vy + vy, vz=earth.vz,
        m=0.0)

sim.integrator = "whfast"
sim.dt = dt

positions = np.zeros((n_steps, sim.N, 2))  # solo plano XY
for i in range(n_steps):
    sim.integrate(sim.t + dt)
    for j, p in enumerate(sim.particles):
        positions[i, j, 0] = p.x
        positions[i, j, 1] = p.y

# ------------------------------------------------------------------------------
# ANIMACIÓN DE LA TRAYECTORIA HACIA ATRÁS
# ------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 8))
colors = ['yellow', 'gray', 'orange', 'blue', 'red', 'brown', 'gold', 'cyan', 'blueviolet', 'black']
lines = [ax.plot([], [], color=colors[i], lw=1)[0] for i in range(len(planet_names))]
points = [ax.plot([], [], 'o', color=colors[i], label=planet_names[i])[0] for i in range(len(planet_names))]

ax.set_xlim(-35, 35)
ax.set_ylim(-35, 35)
ax.set_xlabel("Distance (AU)")        # Unidad en eje X
ax.set_ylabel("Distance (AU)")       # Unidad en eje Y
ax.set_aspect('equal')
ax.legend(loc='upper right')
year_text = ax.text(0.5, 1.02, '', transform=ax.transAxes,
                    ha='center', va='bottom', fontsize=14, fontweight='bold')

def update(frame):
    for i in range(sim.N):
        x = positions[:frame, i, 0]
        y = positions[:frame, i, 1]
        lines[i].set_data(x, y)
        points[i].set_data([positions[frame-1, i, 0]], [positions[frame-1, i, 1]])
    current_fraction = frame / n_steps
    current_year = end_year - current_fraction * years_back
    year_text.set_text(f"Year: {current_year:.1f}")
    return lines + points + [year_text]

ani = FuncAnimation(fig, update, frames=range(10, n_steps, 10), interval=30, blit=False)
plt.show()

# ------------------------------------------------------------------------------
# COMPARACIÓN DE VARIACIÓN DE ENERGÍA ENTRE WHFAST E IAS15
# ------------------------------------------------------------------------------
def run_simulation(integrator_name):
    sim = rebound.Simulation()
    sim.units = ('AU', 'yr', 'Msun')
    sim.add(["Sun", "Mercury", "Venus", "Earth", "Mars",
             "Jupiter", "Saturn", "Uranus", "Neptune"], date=f"{end_year}-01-01")

    earth = sim.particles[3]
    v_kms = random.uniform(15, 45)
    v_auyr = v_kms / 4.74047
    angle = random.uniform(0, 2*np.pi)
    vx = -v_auyr * np.cos(angle)
    vy = -v_auyr * np.sin(angle)

    # 🔀 Aplicar desplazamiento SOLO si usamos IAS15
    x_sat = earth.x + 1e-6 if integrator_name == "ias15" else earth.x

    # Ahora sí, añadir el satélite correctamente
    sim.add(x=x_sat, y=earth.y, z=earth.z,
            vx=earth.vx + vx, vy=earth.vy + vy, vz=earth.vz,
            m=0.0)

    sim.integrator = integrator_name
    if integrator_name == "ias15":
        sim.dt = 1e-1

    energies, times = [], []
    for _ in range(n_steps):
        sim.integrate(sim.t + dt)

        if _ % 1 == 0:
            print(f"IAS15 paso {_}/{n_steps}  tiempo = {sim.t}")

        # Cálculo manual de la energía total
        E_kin = sum(0.5 * p.m * (p.vx**2 + p.vy**2 + p.vz**2) for p in sim.particles if p.m > 0)
        E_pot = 0.0
        for i, pi in enumerate(sim.particles):
            for pj in sim.particles[i+1:]:
                if pi.m > 0 and pj.m > 0:
                    dx, dy, dz = pi.x - pj.x, pi.y - pj.y, pi.z - pj.z
                    r = np.sqrt(dx**2 + dy**2 + dz**2)
                    E_pot -= pi.m * pj.m / r
        energies.append(E_kin + E_pot)
        times.append(sim.t)

    return np.array(times), np.array(energies)


times_whfast, energies_whfast = run_simulation("whfast")
times_ias15, energies_ias15 = run_simulation("ias15")
times_saba4, energies_saba4 = run_simulation("saba4")

plt.figure(figsize=(10, 5))
plt.plot(times_whfast, energies_whfast - energies_whfast[0], label="WHFast", alpha=0.8)
plt.plot(times_ias15, energies_ias15 - energies_ias15[0], label="IAS15", alpha=0.8)
plt.plot(times_saba4, energies_saba4 - energies_saba4[0], label="SABA4", alpha=0.8)
plt.xlabel("Time (years)")
plt.ylabel("Δ Energy (E - E₀)")
plt.title("Variation of energy throughout simulation: Different integration methods")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

"""
results = []

for i in range(Gamma):
    # Copiamos la parte esencial de run_simulation para obtener 'sim'
    sim = rebound.Simulation()
    sim.units = ('AU', 'yr', 'Msun')
    sim.add(["Sun", "Mercury", "Venus", "Earth", "Mars",
             "Jupiter", "Saturn", "Uranus", "Neptune"], date=f"{end_year}-01-01")

    earth = sim.particles[3]
    v_kms = random.uniform(15, 45)
    v_auyr = v_kms / 4.74047
    angle = random.uniform(0, 2 * np.pi)
    vx = -v_auyr * np.cos(angle)
    vy = -v_auyr * np.sin(angle)

    sim.add(x=earth.x, y=earth.y, z=earth.z,
            vx=earth.vx + vx, vy=earth.vy + vy, vz=earth.vz,
            m=0.0)

    sim.integrator = "whfast"
    sim.dt = dt

    for _ in range(n_steps):
        sim.integrate(sim.t + dt)

    asteroid = sim.particles[-1]
    orbit = rebound.Orbit(asteroid, primary=sim.particles[0])

    results.append({
        "asteroid_id": i,
        "integration_years": years_back,
        "final_x": asteroid.x,
        "final_y": asteroid.y,
        "final_z": asteroid.z,
        "a": orbit.a,
        "e": orbit.e,
        "inc": orbit.inc,
        "Omega": orbit.Omega,
        "omega": orbit.omega,
        "f": orbit.f
    })

# Convertir a DataFrame
df = pd.DataFrame(results)
df.to_csv("whfast_asteroid_dataset.csv", index=False)
print("✅ Dataset guardado como 'whfast_asteroid_dataset.csv'")
"""