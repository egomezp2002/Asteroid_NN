
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import random
import rebound
from solar_simulation import simulate_solar_system_hourly
import time  # <-- Agregado para medir tiempo
from datetime import datetime
from multiprocessing import Pool
import os
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from tensorflow import initializers

from scipy.stats import gaussian_kde
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Paso 1: Cargar los datos
csv_name = "asteroides_colisiones_y_no_colisiones.csv"
df = pd.read_csv(csv_name)

# Paso 2: Seleccionar columnas deseadas
cols_deseadas = ['a', 'e', 'i', 'n', 'h', 'label']
df = df[cols_deseadas]

# Paso 3: Calcular percentiles 5 y 95 para 'n' y 'h'
n_p5, n_p95 = df['n'].quantile([0.05, 0.95])
h_p5, h_p95 = df['h'].quantile([0.05, 0.95])

# Paso 4: Filtrar para conservar solo el 90% significativo
df_filtrado = df[
    (df['n'] >= n_p5) & (df['n'] <= n_p95) &
    (df['h'] >= h_p5) & (df['h'] <= h_p95)
]

# Paso 5: Guardar el nuevo CSV
df_filtrado.to_csv("orbitales_filtrados_v2.csv", index=False)
print("✅ CSV guardado: orbitales_filtrados_v2.csv")

parametros = ['n', 'h']
colores = {0: 'tab:blue', 1: 'tab:red'}

for param in parametros:
    plt.figure(figsize=(10, 5))

    # Separar impactores y no impactores
    data0 = df[df['label'] == 0][param].dropna()
    data1 = df[df['label'] == 1][param].dropna()

    # Filtrado del 80% central (para n y h)
    if param in ['n', 'h']:
        combined = pd.concat([data0, data1])
        p10, p90 = np.percentile(combined, [10, 90])
        data0 = data0[(data0 >= p10) & (data0 <= p90)]
        data1 = data1[(data1 >= p10) & (data1 <= p90)]

    # Rango común para KDE
    min_val = min(data0.min(), data1.min())
    max_val = max(data0.max(), data1.max())
    x_vals = np.linspace(min_val, max_val, 500)

    # Histogramas
    plt.hist(data0, bins=50, alpha=0.4, color=colores[0], label='Non-impactor (0)', density=True)
    plt.hist(data1, bins=50, alpha=0.4, color=colores[1], label='Impactor (1)', density=True)

    # KDEs
    if len(data0) > 1:
        kde0 = gaussian_kde(data0)
        plt.plot(x_vals, kde0(x_vals), color=colores[0], lw=2)
    if len(data1) > 1:
        kde1 = gaussian_kde(data1)
        plt.plot(x_vals, kde1(x_vals), color=colores[1], lw=2)

    # Título adaptado según filtro
    titulo = f"Distribution of '{param}'"
    if param in ['n', 'h']:
        titulo += " (80% central)"
    
    plt.title(titulo)
    plt.xlabel(param)
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Cargar el CSV filtrado
df = pd.read_csv("orbitales_filtrados_v2.csv")

columnas_a_normalizar = ['a', 'e', 'i', 'n', 'h']
scaler = StandardScaler()

# Aplicar scaler a esas columnas
df_scaled = scaler.fit_transform(df[columnas_a_normalizar])

# Convertir de nuevo a DataFrame
df_norm = pd.DataFrame(df_scaled, columns=columnas_a_normalizar)

# Agregar la columna 'label' sin modificar
df_norm['label'] = df['label'].values

# Guardar como CSV si lo deseas
df_norm.to_csv("orbitales_estandarizados_v2.csv", index=False)
print("✅ CSV estandarizado guardado como: orbitales_estandarizados_v2.csv")

import tensorflow as tf
from sklearn.model_selection import train_test_split

# Seleccionar las columnas de entrada y la salida (label)
# X = df[['a', 'e', 'i', 'Omega', 'omega', 'n', 'h']].values
X = df_norm[['a', 'e', 'i', 'n', 'h']].values
y = df_norm['label'].values

# Convertir etiquetas: 1 (KI) → 0.9, 0 (observado) → 0.1
# Etiquetas suaves: 0.9 para KIs (impactadores probables), 0.1 para objetos observados (muy improbables): es una recomendación del paper
y = np.where(y == 1, 1, 0)

# Separar datos: 90% entrenamiento / 10% prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

#Comprobaciones propuestas por chatgpt: por qué no funciona?

print(np.unique(y_train, return_counts=True))
print(np.std(X_train, axis=0))


# Crear el modelo HOI
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(5,)),
    tf.keras.layers.Dense(7, activation='relu', kernel_initializer='he_normal'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(3, activation='relu', kernel_initializer='he_normal'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer='glorot_uniform')
])

# Compilar el modelo con Adam más agresivo
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=[
                  tf.keras.metrics.MeanAbsoluteError(name='mae'),
                  tf.keras.metrics.AUC(name='auc'),
                  tf.keras.metrics.BinaryAccuracy(name='bin_acc', threshold=0.5)
              ])

# Entrenar el modelo con validación
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)

y_pred = model.predict(X_test[:10])
print("Predicciones del modelo:", y_pred.ravel())

# Guardar el modelo entrenado
model.save("modelo_entrenado_v2.keras")
print("✅ Modelo guardado como modelo_entrenado_v2.keras")

## Con modelo guardado, no es necesario entrenar CADA vez, sino que se reutiliza el modelo

from keras.models import load_model
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import random
import rebound
from solar_simulation import simulate_solar_system_hourly
import time  # <-- Agregado para medir tiempo
from datetime import datetime
from multiprocessing import Pool
import os
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from tensorflow import initializers

model = load_model("modelo_entrenado_v2.keras")

X = df_norm[['a', 'e', 'i', 'n', 'h']].values
y = df_norm['label'].values
y = np.where(y == 1, 1, 0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


# Evaluar en el conjunto de prueba
loss, mae, accuracy, bin_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"📊 Test Loss: {loss:.4f} | MAE: {mae:.4f} | AUC: {accuracy:.4f} | Binary Accuracy: {bin_acc:.4f}")

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

X = df_norm[['a', 'e', 'i', 'n', 'h']].values

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap="coolwarm", alpha=0.6)
plt.title("Reducción PCA de datos normalizados")
plt.xlabel("Componente 1")
plt.ylabel("Componente 2")
plt.colorbar(label="Label (0=No colisión, 1=Colisión)")
plt.show()

