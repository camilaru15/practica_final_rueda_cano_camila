"""
=============================================================================
PRÁCTICA FINAL — EJERCICIO 2
Inferencia y Modelado
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Crear carpeta output
os.makedirs("output", exist_ok=True)


# =============================================================================
# CARGA Y PREPROCESAMIENTO
# =============================================================================

def cargar_datos(ruta):
    return pd.read_csv(ruta)


def preprocesar_datos(df, target):
    """
    Limpieza + encoding
    """
    # Eliminar columnas con muchos nulos (>50%)
    df = df.loc[:, df.isnull().mean() < 0.5]

    # Separar variables
    X = df.drop(columns=[target])
    y = df[target]

    # One-hot encoding
    X = pd.get_dummies(X, drop_first=True)

    # Rellenar nulos restantes con la media
    X = X.fillna(X.mean())

    return X, y


# =============================================================================
# MODELADO
# =============================================================================

def entrenar_modelo(X, y):
    """
    Divide datos, escala y entrena modelo
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    return modelo, X_test, y_test


# =============================================================================
# EVALUACIÓN
# =============================================================================

def evaluar_modelo(modelo, X_test, y_test):
    """
    Calcula métricas y guarda resultados
    """
    y_pred = modelo.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    with open("output/ej2_metricas.txt", "w") as f:
        f.write(f"MAE: {mae}\n")
        f.write(f"RMSE: {rmse}\n")
        f.write(f"R2: {r2}\n")

    print("\n--- MÉTRICAS ---")
    print(f"MAE: {mae}")
    print(f"RMSE: {rmse}")
    print(f"R2: {r2}")

    return y_pred


# =============================================================================
# RESIDUOS
# =============================================================================

def graficar_residuos(y_test, y_pred):
    """
    Gráfico de residuos
    """
    residuos = y_test - y_pred

    plt.scatter(y_pred, residuos)
    plt.axhline(y=0)
    plt.xlabel("Predicciones")
    plt.ylabel("Residuos")
    plt.title("Gráfico de residuos")

    plt.savefig("output/ej2_residuos.png")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    DATA_PATH = "data/dataset.csv"
    TARGET = "SalePrice"

    df = cargar_datos(DATA_PATH)

    X, y = preprocesar_datos(df, TARGET)

    modelo, X_test, y_test = entrenar_modelo(X, y)

    y_pred = evaluar_modelo(modelo, X_test, y_test)

    graficar_residuos(y_test, y_pred)

    print("\n[OK] Ejercicio 2 completado")

