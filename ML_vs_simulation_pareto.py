import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ===================== CONFIGURACIÓN GLOBAL =====================
YEARS = 300               # pon 300 si quieres; para pruebas rápidas, 50–100
dts    = [0.02, 0.01, 0.005]           # REBOUND/WHFast: pasos (años)
rtols  = [1e-6, 1e-8, 1e-10]           # solve_ivp/RK45: tolerancias (usaremos atol=rtol)

# Masas (Msun)
M_sun, M_earth, M_jup = 1.0, 3.0034896e-6, 9.5458e-4

# Asteroide (keplerianos de ejemplo)
a_ast, e_ast = 2.5, 0.30
inc_ast, Omega_ast, omega_ast, M_ast = np.radians(5.0), 0.0, 0.0, 0.0

# ===================== UTILIDADES =====================
def compute_energy(y, masses):
    """ Energía total para N cuerpos (G=1). y: [x... y... z..., vx... vy... vz...] """
    N = len(masses)
    pos = y[:3*N].reshape(N,3)
    vel = y[3*N:].reshape(N,3)
    KE = 0.5*np.sum(masses[:,None]*np.sum(vel**2, axis=1, keepdims=True))
    PE = 0.0
    for i in range(N):
        for j in range(i+1, N):
            rij = np.linalg.norm(pos[i]-pos[j])
            PE -= masses[i]*masses[j]/rij
    return (KE+PE).item()

def nbody_rhs(t, y, masses):
    """ Derivadas Newtonianas (G=1). """
    N = len(masses)
    pos = y[:3*N].reshape(N,3)
    vel = y[3*N:].reshape(N,3)
    acc = np.zeros_like(pos)
    for i in range(N):
        for j in range(N):
            if i == j: continue
            r = pos[j]-pos[i]
            r3 = np.linalg.norm(r)**3
            acc[i] += masses[j]*r/r3
    return np.concatenate([vel.reshape(-1), acc.reshape(-1)])

def build_initial_states_with_rebound():
    import rebound
    sim = rebound.Simulation()
    sim.units = ('AU','yr','Msun')
    sim.G = 1.0
    sim.add(m=M_sun)                           # Sol
    sim.add(m=M_earth,   a=1.0,    e=0.0167)   # Tierra
    sim.add(m=M_jup,     a=5.2044, e=0.0489)   # Júpiter
    sim.add(m=0.0, a=a_ast, e=e_ast, inc=inc_ast, Omega=Omega_ast, omega=omega_ast, M=M_ast)  # asteroide (m=0)
    sim.move_to_com()
    return sim

def pack_state_from_rebound(sim):
    parts = sim.particles
    N = len(parts)
    y0 = np.zeros(6*N)
    for i,p in enumerate(parts):
        y0[3*i:3*i+3] = [p.x, p.y, p.z]
        y0[3*N+3*i:3*N+3*i+3] = [p.vx, p.vy, p.vz]
    masses = np.array([p.m for p in parts])
    return y0, masses

# ===================== CORRIDAS: REBOUND WHFAST =====================
def run_rebound_whfast(dt, years=YEARS):
    import rebound
    sim = build_initial_states_with_rebound()
    sim.integrator = "whfast"
    sim.dt = dt
    # sin correctores para máxima velocidad
    if hasattr(sim, "ri_whfast"):
        sim.ri_whfast.corrector = 0

    E0 = sim.energy()
    t0 = time.perf_counter()
    sim.integrate(years)
    wall = time.perf_counter() - t0
    Ef = sim.energy()
    rel_err = abs(Ef - E0)/abs(E0)
    return wall, rel_err

# ===================== CORRIDAS: solve_ivp RK45 =====================
def run_ivp_rk45(rtol, years=YEARS):
    # mismas IC que REBOUND
    sim0 = build_initial_states_with_rebound()
    y0, masses = pack_state_from_rebound(sim0)

    # Integra 0..years con tolerancias (atol=rtol); dejamos que sea adaptativo
    t0 = time.perf_counter()
    sol = solve_ivp(
        fun=lambda t,y: nbody_rhs(t,y,masses),
        t_span=(0.0, years),
        y0=y0,
        method='RK45',
        rtol=rtol, atol=rtol
    )
    wall = time.perf_counter() - t0
    if not sol.success:
        # En caso de fallo, devolver NaN para excluir del Pareto
        return wall, np.nan

    E0 = compute_energy(y0, masses)
    Ef = compute_energy(sol.y[:,-1], masses)
    rel_err = abs(Ef - E0)/abs(E0)
    return wall, rel_err

# ===================== BARRIDOS Y GRÁFICO PARETO =====================
reb_cost, reb_err = [], []
for dt in dts:
    w,e = run_rebound_whfast(dt)
    reb_cost.append(w); reb_err.append(e)
    print(f"REBOUND WHFast dt={dt}: time={w:.3f}s, |ΔE/E|={e:.3e}")

ivp_cost, ivp_err = [], []
for r in rtols:
    w,e = run_ivp_rk45(r)
    ivp_cost.append(w); ivp_err.append(e)
    print(f"solve_ivp RK45 rtol={r:g}: time={w:.3f}s, |ΔE/E|={e:.3e}")

# ===================== PLOT (Coste vs Precisión) =====================
plt.figure(figsize=(8.5,6))
plt.plot(reb_cost, reb_err, 'o-', lw=2, label='REBOUND WHFast (var. $\\Delta t$)')
for c,e,dt in zip(reb_cost, reb_err, dts):
    plt.annotate(f"dt={dt}", (c,e), textcoords="offset points", xytext=(6,6), fontsize=12, weight='bold')

plt.plot(ivp_cost, ivp_err, 's--', lw=2, label='solve_ivp RK45 (var. tol)')
for c,e,r in zip(ivp_cost, ivp_err, rtols):
    plt.annotate(f"rtol={r:g}", (c,e), textcoords="offset points", xytext=(6,-14), fontsize=12, weight='bold')

plt.yscale('log')
plt.xlabel("Wall time (s)", fontsize=18, fontweight='bold')
plt.ylabel("Final relative energy error |ΔE/E|", fontsize=18, fontweight='bold')
plt.title(f"Cost–Accuracy Trade-off (Pareto)\nSun–Earth–Jupiter–Asteroid, T={YEARS} yr", fontsize=20, fontweight='bold')
leg = plt.legend(fontsize=14)
for t in leg.get_texts(): t.set_fontweight('bold')
plt.tick_params(axis='both', labelsize=14)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
