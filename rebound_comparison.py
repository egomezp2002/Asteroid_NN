
import rebound
import numpy as np
import time
import matplotlib.pyplot as plt

# Parámetros del sistema
G = 4 * np.pi**2  # Constante gravitacional en AU^3 / (yr^2 * Msol)

# Función para calcular energía total
def calcular_energia_total(sim):
    energia_cinetica = 0.0
    energia_potencial = 0.0
    for i, p1 in enumerate(sim.particles):
        energia_cinetica += 0.5 * p1.m * (p1.vx**2 + p1.vy**2 + p1.vz**2)
        for j, p2 in enumerate(sim.particles):
            if j > i:
                dx, dy, dz = p1.x - p2.x, p1.y - p2.y, p1.z - p2.z
                r = np.sqrt(dx**2 + dy**2 + dz**2)
                energia_potencial -= G * p1.m * p2.m / r
    return energia_cinetica + energia_potencial

# Parámetros comunes del sistema solar reducido
particulas = [
    {"name": "Sun", "m": 1.0, "a": 0},
    {"name": "Earth", "m": 3e-6, "a": 1.0},
    {"name": "Jupiter", "m": 9.5e-4, "a": 5.2},
    {"name": "Saturn", "m": 2.85e-4, "a": 9.5},
    {"name": "Asteroid", "m": 0.0, "a": 4.0}
]

metodos = ["ias15", "saba4", "whfast"]
resultados = {}

for metodo in metodos:
    sim = rebound.Simulation()
    sim.G = G
    sim.units = ("AU", "yr", "Msun")
    sim.integrator = metodo
    
    # Añadir partículas
    sim.add(m=1.0)              # Sol
    sim.add(m=1.65e-7, a=0.387)  # Mercurio
    sim.add(m=2.45e-6, a=0.723)  # Venus
    sim.add(m=3.00e-6, a=1.000)  # Tierra
    sim.add(m=3.21e-7, a=1.524)  # Marte
    sim.add(m=9.55e-4, a=5.204)  # Júpiter
    sim.add(m=2.85e-4, a=9.583)  # Saturno
    sim.add(m=4.37e-5, a=19.18)  # Urano
    sim.add(m=5.15e-5, a=30.07)  # Neptuno
    sim.add(m=0.0, a=4.0)        # Asteroide



    sim.move_to_com()

    energia_inicial = calcular_energia_total(sim)
    pasos = 50000
    dt = 200.0 / pasos  # 10 años

    inicio = time.time()
    for paso in range(pasos):
        sim.integrate(sim.t + dt)
        print(f"paso {paso + 1} del método {metodo.upper()}")
    duracion = time.time() - inicio

    energia_final = calcular_energia_total(sim)
    perdida_energia = abs((energia_final - energia_inicial) / energia_inicial)

    resultados[metodo] = {
        "energia_inicial": energia_inicial,
        "energia_final": energia_final,
        "perdida_energia": perdida_energia,
        "tiempo_simulacion": duracion
    }

# Mostrar resultados
print("\n--- RESULTADOS COMPARATIVOS ---")
for metodo in metodos:
    r = resultados[metodo]
    print(f"\nMétodo: {metodo.upper()}")
    print(f"  Energía inicial: {r['energia_inicial']:.5e}")
    print(f"  Energía final:   {r['energia_final']:.5e}")
    print(f"  Pérdida relativa de energía: {r['perdida_energia']:.5e}")
    print(f"  Tiempo de simulación: {r['tiempo_simulacion']:.3f} segundos")

# Extraer datos del diccionario resultados
metodos = list(resultados.keys())
perdida_energia = [resultados[m]["perdida_energia"] for m in metodos]
tiempos_simulacion = [resultados[m]["tiempo_simulacion"] for m in metodos]

# Gráfico de pérdida de energía
plt.figure()
plt.bar(metodos, perdida_energia)
plt.ylabel("Relative Energy Loss")
plt.title("Energy Conservation (10 bodies, 200 years, 50,000 steps)")
plt.yscale("log")
plt.grid(True)
plt.tight_layout()
plt.savefig("energy_loss_comparison.png", format="png")

# Gráfico de tiempo de simulación
plt.figure()
plt.bar(metodos, tiempos_simulacion)
plt.ylabel("Simulation Time (s)")
plt.title("Computation Time (10 bodies, 200 years, 50,000 steps)")
plt.grid(True)
plt.tight_layout()
plt.savefig("simulation_time_comparison.png", format="png")

plt.show()