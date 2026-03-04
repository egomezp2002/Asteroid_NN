import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Parámetros Nominales
r_nom = np.array([1.0, 0.0, 0.0])    
v_nom = np.array([0.0, 2 * np.pi, 0.0])  # Órbita circular exacta en UA/año

# 2. Matriz de Rotación C (Inercial -> RTN)
u_R = r_nom / np.linalg.norm(r_nom)
h = np.cross(r_nom, v_nom)
u_N = h / np.linalg.norm(h)
u_T = np.cross(u_N, u_R)
C = np.array([u_R, u_T, u_N])

# 3. Monte Carlo
n = 4000
sigmapos = 1e-5
# Importante: Generamos ruido independiente en cada eje para que sea una esfera
ruido_xyz = np.random.normal(0, sigmapos, (n, 3))

# 4. Transformación
# Aquí multiplicamos cada vector de ruido por la matriz C
pos_rtn = np.dot(ruido_xyz, C.T) 

# 5. Visualización 3D Pro
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

R, T, N = pos_rtn[:, 0], pos_rtn[:, 1], pos_rtn[:, 2]

sc = ax.scatter(R, T, N, c=np.linalg.norm(pos_rtn, axis=1), cmap='magma', s=3, alpha=0.5)

# --- EL TRUCO PARA QUE SEA UNA ESFERA ---
# Esto obliga a que 1 unidad en X sea igual a 1 unidad en Y y Z visualmente
ax.set_aspect('equal') 

ax.set_xlabel('Radial (R) [UA]')
ax.set_ylabel('Tangencial (T) [UA]')
ax.set_zlabel('Normal (N) [UA]')
ax.set_title('Nube de Incertidumbre Isótropa en RTN')

plt.colorbar(sc, label='Distancia al centro')
plt.show()