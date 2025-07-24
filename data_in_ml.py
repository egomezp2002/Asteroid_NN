
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

cols_deseadas = ['a', 'e', 'i', 'Omega', 'omega', 'nu', 'mean_speed', 'h', 'año_epoca', 'label']
df = df[cols_deseadas]

df.to_csv("orbitales_filtrados.csv", index=False)
print("✅ CSV guardado: orbitales_filtrados.csv")

# Cargar el CSV filtrado
df = pd.read_csv("orbitales_filtrados.csv")

# Columnas a normalizar (excluyendo 'label' y 'año_epoca')
columnas_a_normalizar = ['a', 'e', 'i', 'Omega', 'omega', 'nu', 'mean_speed', 'h']

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

# No tocar estas dos columnas
df_norm['año_epoca'] = df['año_epoca']
df_norm['label'] = df['label']

# Guardar el nuevo CSV
df_norm.to_csv("orbitales_normalizados.csv", index=False)
print("✅ CSV normalizado guardado como: orbitales_normalizados.csv")

import tensorflow as tf
from sklearn.model_selection import train_test_split

# Seleccionar las columnas de entrada y la salida (label)
# X = df[['a', 'e', 'i', 'Omega', 'omega', 'nu', 'h']].values
X = df[['a', 'e', 'i', 'Omega', 'omega']].values
y = df['label'].values

# Separar en datos de entrenamiento y prueba (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Crear el modelo
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(X.shape[1],)),  # Capa oculta 1
    tf.keras.layers.Dense(8, activation='relu'),                               # Capa oculta 2
    tf.keras.layers.Dense(1, activation='sigmoid')                             # Capa de salida (binaria)
])

# Compilar el modelo
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Entrenar el modelo
model.fit(X_train, y_train, epochs=500, batch_size=16, validation_split=0.2)

# Evaluar en el conjunto de prueba
loss, accuracy = model.evaluate(X_test, y_test)
print(f"🔍 Precisión en test: {accuracy:.4f}")