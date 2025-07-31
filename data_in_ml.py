
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

csv_name = "asteroides_colisiones_y_no_colisiones.csv"  # Ajusta según tu caso
df = pd.read_csv(csv_name)

cols_deseadas = ['a', 'e', 'i', 'Omega', 'omega', 'label']
df = df[cols_deseadas]

df.to_csv("orbitales_filtrados.csv", index=False)
print("✅ CSV guardado: orbitales_filtrados.csv")

# Cargar el CSV filtrado
df = pd.read_csv("orbitales_filtrados.csv")

columnas_a_normalizar = ['a', 'e', 'i', 'Omega', 'omega']
scaler = StandardScaler()

# Aplicar scaler a esas columnas
df_scaled = scaler.fit_transform(df[columnas_a_normalizar])

# Convertir de nuevo a DataFrame
df_norm = pd.DataFrame(df_scaled, columns=columnas_a_normalizar)

# Agregar la columna 'label' sin modificar
df_norm['label'] = df['label'].values

# Guardar como CSV si lo deseas
df_norm.to_csv("orbitales_estandarizados.csv", index=False)
print("✅ CSV estandarizado guardado como: orbitales_estandarizados.csv")

import tensorflow as tf
from sklearn.model_selection import train_test_split

# Seleccionar las columnas de entrada y la salida (label)
# X = df[['a', 'e', 'i', 'Omega', 'omega', 'nu', 'h']].values
X = df_norm[['a', 'e', 'i', 'Omega', 'omega']].values
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
model.save("modelo_entrenado.keras")
print("✅ Modelo guardado como modelo_entrenado.keras")

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

model = load_model("modelo_entrenado.keras")

X = df_norm[['a', 'e', 'i', 'Omega', 'omega']].values
y = df_norm['label'].values
y = np.where(y == 1, 1, 0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


# Evaluar en el conjunto de prueba
loss, mae, accuracy, bin_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"📊 Test Loss: {loss:.4f} | MAE: {mae:.4f} | AUC: {auc:.4f} | Binary Accuracy: {bin_acc:.4f}")

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

X = df_norm[['a', 'e', 'i', 'Omega', 'omega']].values

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap="coolwarm", alpha=0.6)
plt.title("Reducción PCA de datos normalizados")
plt.xlabel("Componente 1")
plt.ylabel("Componente 2")
plt.colorbar(label="Label (0=No colisión, 1=Colisión)")
plt.show()

