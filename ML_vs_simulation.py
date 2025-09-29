
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ========= PARÁMETROS =========
YEARS = 300          # cámbialo a 300 cuando quieras
dt     = 0.001       # años (~3.65 días)
SAVE_ENERGY_EVERY = 10   # guarda energía cada N pasos para no penalizar en exceso

# ========= MASAS (M_sun) =========
M_sun     = 1.0
M_earth   = 3.0034896e-6
M_jupiter = 9.5458e-4

# ========= IC DEL ASTEROIDE (keplerianos aprox) =========
a_ast = 2.5     # AU
e_ast = 0.30
inc_ast = np.radians(5.0)
Omega_ast = 0.0
omega_ast = 0.0
M_ast     = 0.0  # anomalía media

# ========= UTILIDADES =========
def compute_energy(y, masses):
    """
    Energía total (cinética + potencial) para N cuerpos en unidades G=1.
    y: vector [x1..xN, y1..yN, z1..zN, vx1.., vy1.., vz1..] tamaño 6N
    masses: array (N,)
    """
    N = len(masses)
    pos = y[:3*N].reshape(N,3)
    vel = y[3*N:].reshape(N,3)

    # Cinética
    KE = 0.5 * np.sum(masses[:,None] * np.sum(vel**2, axis=1, keepdims=True))

    # Potencial
    PE = 0.0
    for i in range(N):
        for j in range(i+1, N):
            rij = np.linalg.norm(pos[i]-pos[j])
            PE -= masses[i]*masses[j]/rij
    return (KE + PE).item()

def nbody_rhs(t, y, masses):
    """ Derivadas para N-cuerpos (aceleraciones newtonianas, G=1). """
    N = len(masses)
    pos = y[:3*N].reshape(N,3)
    vel = y[3*N:].reshape(N,3)
    acc = np.zeros_like(pos)

    for i in range(N):
        # asteroide puede tener masa 0 (no afecta a otros)
        for j in range(N):
            if i == j: 
                continue
            r = pos[j] - pos[i]
            r3 = (np.linalg.norm(r)**3)
            acc[i] += masses[j] * r / r3
    dydt = np.concatenate([vel.reshape(-1), acc.reshape(-1)])
    return dydt

# ========= CONSTRUYE ICs CON REBOUND PARA REUTILIZARLAS EN solve_ivp =========
def build_initial_states_with_rebound():
    import rebound
    sim = rebound.Simulation()
    sim.units = ('AU','yr','Msun')
    sim.G = 1.0
    sim.add(m=M_sun)                         # Sol
    sim.add(m=M_earth,   a=1.0,   e=0.0167)  # Tierra aprox
    sim.add(m=M_jupiter, a=5.2044, e=0.0489) # Júpiter aprox
    sim.add(m=0.0, a=a_ast, e=e_ast, inc=inc_ast, Omega=Omega_ast, omega=omega_ast, M=M_ast)  # asteroide test
    sim.move_to_com()
    return sim

def rebound_run_and_energy(sim, YEARS, dt, save_stride=10):
    import rebound
    sim.integrator = "whfast"
    sim.dt = dt
    # ====== FIX: usar el namespace correcto para parámetros de WHFast ======
    if hasattr(sim, "ri_whfast"):
        sim.ri_whfast.corrector = 0   # 0 = sin correctores; 11/13 si quisieras correctores de orden alto
    # ======================================================================

    # Estado inicial y energía
    E0 = sim.energy()
    times, Erel = [], []
    t_target = 0.0
    nsteps = int(np.floor(YEARS/dt))
    t0 = time.perf_counter()
    for k in range(1, nsteps+1):
        t_target = k*dt
        sim.integrate(t_target)
        if (k % save_stride)==0 or k==nsteps:
            Et = sim.energy()
            times.append(t_target)
            Erel.append((Et - E0)/abs(E0))
    t_elapsed = time.perf_counter() - t0
    return np.array(times), np.array(Erel), t_elapsed, sim

def pack_state_from_rebound(sim):
    # empaqueta (x,y,z,vx,vy,vz) para todos los cuerpos
    parts = sim.particles
    N = len(parts)
    y0 = np.zeros(6*N)
    for i,p in enumerate(parts):
        y0[3*i:3*i+3] = [p.x, p.y, p.z]
        y0[3*N+3*i:3*N+3*i+3] = [p.vx, p.vy, p.vz]
    masses = np.array([p.m for p in parts])
    return y0, masses

# ========= MAIN EXPERIMENTO =========
sim = build_initial_states_with_rebound()

# --- REBOUND (WHFast, paso fijo) ---
t_reb, Erel_reb, time_reb, sim_final = rebound_run_and_energy(sim, YEARS, dt, SAVE_ENERGY_EVERY)

# --- Misma IC para solve_ivp ---
y0, masses = pack_state_from_rebound(sim)

# Mallado de salida (forzamos muestreo uniforme); solve_ivp podrá substepear internamente
t_eval = np.arange(0.0, YEARS + 1e-12, dt)

# Integra con solve_ivp (RK45) limitado por max_step=dt (sigue siendo no-simpléctico)
t0 = time.perf_counter()
sol = solve_ivp(
    fun=lambda t,y: nbody_rhs(t,y,masses),
    t_span=(0.0, YEARS),
    y0=y0,
    method='RK45',
    t_eval=t_eval,
    rtol=1e-9,
    atol=1e-12,
    max_step=dt
)
time_ivp = time.perf_counter() - t0

# Energía relativa para solve_ivp (calculada en los puntos t_eval)
E0_ivp = compute_energy(sol.y[:,0], masses)
Erel_ivp = np.array([ (compute_energy(sol.y[:,i], masses)-E0_ivp)/abs(E0_ivp) for i in range(sol.y.shape[1]) ])
t_ivp = sol.t

# ========= REPORTE BÁSICO =========
# ========= REPORTE BÁSICO =========
# ========= REPORTE BÁSICO =========
print(f"REBOUND WHFast: runtime = {time_reb:.3f} s, samples of energy = {len(t_reb)}")
print(f"solve_ivp RK45: runtime = {time_ivp:.3f} s, nfev = {sol.nfev}, steps = {len(t_ivp)}")

# ========= GRÁFICO DE ENERGÍA =========
plt.figure(figsize=(10, 6))
plt.plot(t_reb, np.abs(Erel_reb), label="REBOUND WHFast (|ΔE/E|)", lw=3)
plt.plot(t_ivp, np.abs(Erel_ivp), label="solve_ivp RK45 (|ΔE/E|)", lw=3, alpha=0.8)

plt.yscale('log')
plt.xlabel("Time (years)", fontsize=20, fontweight='bold')
plt.ylabel("Relative Energy Error |ΔE/E|", fontsize=20, fontweight='bold')
plt.title("Energy Conservation vs. Time (Sun + Earth + Jupiter + Asteroid)", fontsize=24, fontweight='bold')

leg = plt.legend(fontsize=18)
for text in leg.get_texts():
    text.set_fontweight('bold')

plt.tick_params(axis='both', labelsize=18)
plt.grid(True, alpha=0.5, linestyle='--')
plt.tight_layout()
plt.show()

# ========= GRÁFICO DE TIEMPOS =========
plt.figure(figsize=(8, 5))
bars = plt.bar(["REBOUND WHFast", "solve_ivp RK45"], [time_reb, time_ivp],
               color=["tab:blue", "tab:orange"], edgecolor='black')

plt.ylabel("Wall Time (s)", fontsize=20, fontweight='bold')
plt.title(f"Runtime Comparison ({YEARS} years, dt={dt} yr)", fontsize=24, fontweight='bold')

# Añadir valores encima de las barras
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 0.01, f"{height:.2f}s",
             ha='center', va='bottom', fontsize=18, fontweight='bold')

plt.tick_params(axis='both', labelsize=18)
plt.grid(axis='y', alpha=0.5, linestyle='--')
plt.tight_layout()
plt.show()

def angular_momentum(y, m):
    N = len(m); r = y[:3*N].reshape(N,3); v = y[3*N:].reshape(N,3)
    Lvec = (m[:,None]*np.cross(r, v)).sum(axis=0)
    return np.linalg.norm(Lvec)

L0 = angular_momentum(sol.y[:,0], masses)
L_rel_ivp = [abs(angular_momentum(sol.y[:,i], masses)-L0)/abs(L0) for i in range(sol.y.shape[1])]
plt.figure(figsize=(9,5))
plt.plot(t_ivp, L_rel_ivp, lw=2, label='solve_ivp RK45 (|ΔL|/L)')
plt.yscale('log'); plt.xlabel('Time (years)'); plt.ylabel('Relative angular momentum error')
plt.title('Angular momentum conservation'); plt.grid(True, ls='--', alpha=0.5); plt.legend(); plt.show()

plt.show()

# ========= NOTAS:
# - WHFast es simpléctico de paso fijo: conserva integrales a nivel oscilatorio y suele ser más rápido por paso.
# - solve_ivp (RK45) es no-simpléctico y, aun imponiendo max_step=dt, puede substep-ear internamente -> más coste.
# - Sube YEARS a 300 y mantén dt para ver cómo la diferencia de tiempo y la deriva (pequeña) de energía se acentúan.
# - Puedes cambiar WHFast por IAS15 (sim.integrator='ias15') si quieres comparar integrador de orden 15 y adaptativo.

