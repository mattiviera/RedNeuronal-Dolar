## Tensorflow https://www.tensorflow.org/?hl=es-419

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 1. Definición de datos: Dólares y su equivalente en pesos argentinos (1 USD = 1175 ARS)
dolares = np.array([0, 1, 5, 10, 20, 50, 100], dtype=float)
pesos = np.array([0, 1175, 5875, 11750, 23500, 58750, 117500], dtype=float)

# 2. Definición del modelo
print("🔧 Definiendo el modelo...")
capa = tf.keras.layers.Dense(units=1, input_shape=[1])
modelo = tf.keras.Sequential([capa])

# 3. Compilación del modelo
modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)

# 4. Entrenamiento
print("⏳ Entrenando el modelo...")
historial = modelo.fit(dolares, pesos, epochs=1000, verbose=False)
print("✅ Modelo entrenado")

# 5. Visualización del entrenamiento
plt.xlabel("Épocas (época #)")
plt.ylabel("Pérdida (loss)")
plt.plot(historial.history["loss"])
plt.show()

# 6. Prueba del modelo
print("🔍 Probando el modelo con una predicción...")
usd_prueba = 70  # cambiá este valor para probar con otros montos
resultado = modelo.predict(np.array([usd_prueba], dtype=float))
print(f"{usd_prueba} USD ≈ {resultado[0][0]:.2f} ARS (según modelo)")
