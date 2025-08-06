
# evaluar_modelo.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from keras.models import load_model

# 1. Cargar datos
df_norm = pd.read_csv("orbitales_estandarizados_v2.csv")
# 2. Preparar X e y
X = df_norm[['a', 'e', 'i', 'n', 'h']].values
y = np.where(df_norm['label'].values == 1, 1, 0)

# 3. Separar en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# 4. Cargar modelo previamente entrenado
model = load_model("modelo_entrenado_v2.keras")
print("✅ Modelo cargado correctamente")

# 5. Evaluar el modelo
loss, mae, auc, bin_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"📊 Test Loss: {loss:.4f} | MAE: {mae:.4f} | ACCURACY: {auc:.4f} | Binary Accuracy: {bin_acc:.4f}")

# 6. Visualizar con PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="coolwarm", alpha=0.6)
plt.title("PCA Reduction: Normalised Data")
plt.xlabel("Comp 1")
plt.ylabel("Comp 2")
plt.colorbar(scatter, label="Label (0 = Non-impactor, 1 = Impactor)")
plt.tight_layout()
plt.show()
