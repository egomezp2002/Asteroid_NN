
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


csv_name = "asteroides_colisiones_y_no_colisiones.csv"  # Ajusta según tu caso
df = pd.read_csv(csv_name)

cols_deseadas = ['a', 'e', 'i', 'Omega', 'omega', 'n', 'h', 'label']
df = df[cols_deseadas]

df.to_csv("orbitales_filtrados.csv", index=False)
print("✅ CSV guardado: orbitales_filtrados.csv")

# Cargar el CSV filtrado
df = pd.read_csv("orbitales_filtrados.csv")

# Columnas a normalizar (excluyendo 'label' y 'año_epoca')
columnas_a_normalizar = ['a', 'e', 'i', 'Omega', 'omega', 'n', 'h']

# Crear copia del DataFrame para normalizar
df_norm = df.copy()

# Normalizar con fórmula: (x - min) / (max - min)
for col in columnas_a_normalizar:
    col_min = df[col].min()
    col_max = df[col].max()
    if col_max != col_min:
        df_norm[col] = (df[col] - col_min) / (col_max - col_min)
    else:
        df_norm[col] = 0.0  # valor constante

# Las columnas que no se editan
df_norm['label'] = df['label']

# Guardar el nuevo CSV
df_norm.to_csv("orbitales_normalizados.csv", index=False)
print("✅ CSV normalizado guardado como: orbitales_normalizados.csv")

import tensorflow as tf
from sklearn.model_selection import train_test_split

# Seleccionar las columnas de entrada y la salida (label)
# X = df[['a', 'e', 'i', 'Omega', 'omega', 'nu', 'h']].values
X = df_norm[['a', 'e', 'i', 'Omega', 'omega']].values
y = df_norm['label'].values

# Convertir etiquetas: 1 (KI) → 0.9, 0 (observado) → 0.1
# Etiquetas suaves: 0.9 para KIs (impactadores probables), 0.1 para objetos observados (muy improbables): es una recomendación del paper
# y = np.where(y == 1, 0.9, 0.1)

# Separar datos: 90% entrenamiento / 10% prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Crear el modelo HOI
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(5,)),           # Capa de entrada con 5 parámetros
    tf.keras.layers.Dense(7, activation='relu'), # Capa oculta 1 (7 neuronas)
    tf.keras.layers.Dense(3, activation='relu'), # Capa oculta 2 (3 neuronas)
    tf.keras.layers.Dense(1, activation='sigmoid') # Capa de salida binaria
])

# Compilar el modelo con optimizador Adam y entropía cruzada
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=[
                  tf.keras.metrics.MeanAbsoluteError(name='mae'),
                  tf.keras.metrics.AUC(name='auc'),
                  tf.keras.metrics.BinaryAccuracy(name='bin_acc', threshold=0.5)
              ])

# Parada temprana si la mejora por época es < 1%
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    min_delta=0.01,
    restore_best_weights=True
)

# Entrenar el modelo (con validación del 20% de los datos de entrenamiento)
model.fit(X_train, y_train,
          epochs=500,
          batch_size=16,
          validation_split=0.2,
          callbacks=[early_stopping],
          verbose=1)

# Guardar el modelo entrenado
model.save("modelo_entrenado.keras")
print("✅ Modelo guardado como modelo_entrenado.keras")

from keras.models import load_model

model = load_model("modelo_entrenado.keras")

# Evaluar en el conjunto de prueba
loss, mae, auc, bin_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"📊 Test Loss: {loss:.4f} | MAE: {mae:.4f} | AUC: {auc:.4f} | Binary Accuracy: {bin_acc:.4f}")

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap="coolwarm", alpha=0.6)
plt.title("Reducción PCA de datos normalizados")
plt.xlabel("Componente 1")
plt.ylabel("Componente 2")
plt.colorbar(label="Label (0=No colisión, 1=Colisión)")
plt.show()

