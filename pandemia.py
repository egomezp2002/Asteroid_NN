
# -*- coding: utf-8 -*-
"""
SEIRD con demografía, mortalidad por COVID y Rt por tramos — España, inicio 2020-03-10
Ajuste grueso de I0 y E0 al pico de fallecidos (~950 el 2020-04-02).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import gamma as gamma_func

# ---------- Parámetros (fuentes citadas arriba) ----------
nu = 0.0072 / 365.0         # natalidad diaria
mu = 0.0104 / 365.0         # mortalidad natural diaria
incub_days = 5.2            # incubación media -> sigma
infec_days = 7.0            # infeccioso medio -> gamma
IFR = 0.011                 # IFR (exceso) ENE-COVID
mean_inf_to_death = 23.9    # media infección->muerte (días)

sigma = 1.0 / incub_days
gamma = 1.0 / infec_days
delta = IFR/(1.0-IFR) * (gamma + mu)   # fija IFR objetivo

# Rt por tramos (medidas)
rt_schedule = [
    (pd.Timestamp('2020-03-10'), 5.9),   # pre-lockdown
    (pd.Timestamp('2020-03-14'), 1.86),  # estado de alarma
    (pd.Timestamp('2020-03-30'), 0.48),  # "hibernación" no esenciales
    (pd.Timestamp('2020-04-13'), 0.80),  # sigue bajo
    (pd.Timestamp('2020-05-11'), 1.10),  # desescalada
    (pd.Timestamp('2020-06-21'), 1.00),  # fin EDA 1ª ola
]

# Población y condiciones iniciales
N0 = 47_679_489.0
D0 = 35.0
R0_init = 0.0

def rt_of_date(date):
    val = rt_schedule[0][1]
    for start, v in rt_schedule:
        if date >= start:
            val = v
        else:
            break
    return val

def beta_of_date(date):
    return rt_of_date(date) * (gamma + mu + delta)

def seird_step(S, E, I, R, D, beta, dt):
    N = S + E + I + R
    new_inf = beta * S * I / max(N, 1.0)
    dS = nu*N - new_inf - mu*S
    dE = new_inf - sigma*E - mu*E
    dI = sigma*E - (gamma+delta+mu)*I
    dR = gamma*I - mu*R
    dD = delta*I
    return S + dS*dt, E + dE*dt, I + dI*dt, R + dR*dt, D + dD*dt, new_inf

def run_model(E0, I0, t0='2020-03-10', tf='2020-08-01', dt=0.25):
    S0 = N0 - E0 - I0 - R0_init - D0
    S, E, I, R, D = S0, E0, I0, R0_init, D0
    dates = pd.date_range(pd.Timestamp(t0), pd.Timestamp(tf), freq=f'{int(dt*24*60)}T')
    out = {'date':[], 'S':[], 'E':[], 'I':[], 'R':[], 'D':[], 'new_inf':[]}
    for d in dates:
        b = beta_of_date(d)
        S,E,I,R,D,new_inf = seird_step(S,E,I,R,D,b,dt)
        out['date'].append(d); out['S'].append(S); out['E'].append(E)
        out['I'].append(I); out['R'].append(R); out['D'].append(D); out['new_inf'].append(new_inf)
    df = pd.DataFrame(out)
    daily = df.groupby(df['date'].dt.date).agg({'S':'last','E':'last','I':'last','R':'last','D':'last','new_inf':'sum'}).reset_index()
    daily['date'] = pd.to_datetime(daily['date'])
    return df, daily

def deaths_from_convolution(daily, IFR=IFR, mean=mean_inf_to_death, sd=9.0):
    # Kernel gamma discreto
    k = (mean/sd)**2
    theta = sd**2/mean
    t = np.arange(0, 90)
    pdf = (t**(k-1) * np.exp(-t/theta)) / (gamma_func(k) * (theta**k))
    pdf = pdf / pdf.sum()
    deaths = np.convolve(daily['new_inf'].values, IFR * pdf, mode='full')[:len(daily)]
    idx = int(np.argmax(deaths))
    return deaths, daily['date'].iloc[idx], deaths[idx]

# --- Ajuste grueso de I0 y E0 para el pico de fallecidos del 2/abr (~950) ---
I0_grid = np.arange(5_000, 35_000+1, 2_500)
E0_grid = np.arange(10_000, 90_000+1, 5_000)

best = None
for I0 in I0_grid:
    for E0 in E0_grid:
        _, d = run_model(E0, I0)
        deaths, pdate, pval = deaths_from_convolution(d)
        # coste: magnitud + distancia en fecha (ponderada)
        err_mag = (pval - 950.0)**2
        err_t = ((pdate - pd.Timestamp('2020-04-02')).days)**2 * 200.0
        cost = err_mag + err_t
        if best is None or cost < best[0]:
            best = (cost, E0, I0, d, deaths, pdate, pval)

_, E0_hat, I0_hat, daily, deaths_conv, peak_date, peak_val = best
print(f"I0*={I0_hat:,.0f}, E0*={E0_hat:,.0f} | Pico ≈ {peak_val:,.0f} el {peak_date.date()}")

# --- Figuras ---
plt.figure(figsize=(10,6))
plt.plot(daily['date'], daily['S'], label='S')
plt.plot(daily['date'], daily['E'], label='E')
plt.plot(daily['date'], daily['I'], label='I')
plt.plot(daily['date'], daily['R'], label='R')
plt.plot(daily['date'], daily['D'], label='D acumulado')
for d, _ in rt_schedule: plt.axvline(d, linestyle='--', alpha=0.7)
plt.xlabel('Fecha'); plt.ylabel('Personas'); plt.title('SEIRD España 2020'); plt.legend(); plt.tight_layout(); plt.show()

plt.figure(figsize=(10,5))
plt.plot(daily['date'], daily['new_inf'], label='Infecciones/día (modelo)')
plt.plot(daily['date'], deaths_conv, label='Fallecidos/día (conv.)')
for d, _ in rt_schedule: plt.axvline(d, linestyle='--', alpha=0.7)
plt.xlabel('Fecha'); plt.ylabel('Personas/día'); plt.title('Incidencia y fallecimientos diarios'); plt.legend(); plt.tight_layout(); plt.show()
