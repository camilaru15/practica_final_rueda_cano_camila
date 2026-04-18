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
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    confusion_matrix
)
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


# # =============================================================================
# MODELO DE REGRESIÓN LINEAL
# =============================================================================

def modelo_regresion(X, y):
    """
    Entrena modelo de regresión y genera outputs
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)

    # Métricas
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    with open("output/ej2_metricas_regresion.txt", "w") as f:
        f.write(f"MAE: {mae}\n")
        f.write(f"RMSE: {rmse}\n")
        f.write(f"R2: {r2}\n")

    print("\n--- REGRESIÓN LINEAL ---")
    print(f"MAE: {mae}")
    print(f"RMSE: {rmse}")
    print(f"R2: {r2}")

    # Coeficientes
    coef = pd.Series(modelo.coef_, index=X.columns)
    top_coef = coef.abs().sort_values(ascending=False).head(10)

    plt.figure()
    top_coef.sort_values().plot(kind="barh")
    plt.title("Top 10 coeficientes")
    plt.savefig("output/ej2_coeficientes.png")
    plt.close()

    # Residuos
    residuos = y_test - y_pred

    plt.figure()
    plt.scatter(y_pred, residuos)
    plt.axhline(0)
    plt.xlabel("Predicciones")
    plt.ylabel("Residuos")
    plt.title("Gráfico de residuos")
    plt.savefig("output/ej2_residuos.png")
    plt.close()

    return X, y


# =============================================================================
# MODELO DE REGRESIÓN LOGÍSTICA
# =============================================================================

def modelo_logistico(X, y):
    """
    Clasificación por rangos de precio
    """
    # Crear categorías del target
    y_cat = pd.qcut(y, q=4, labels=["bajo", "medio-bajo", "medio-alto", "alto"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    modelo = LogisticRegression(max_iter=1000)
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)

    acc = ...
    prec = ...
    rec = ...
    f1 = ...
    
    with open("output/ej2_metricas_logistica.txt", "w") as f:
        f.write(f"Accuracy: {acc}\n")
        f.write(f"Precision: {prec}\n")
        f.write(f"Recall: {rec}\n")
        f.write(f"F1: {f1}\n")

    print("\n--- REGRESIÓN LOGÍSTICA ---")
    print(f"Accuracy: {acc}")
    print(f"Precision: {prec}")
    print(f"Recall: {rec}")
    print(f"F1: {f1}")

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Matriz de confusión")
    plt.savefig("output/ej2_matriz_confusion.png")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    DATA_PATH = "data/dataset.csv"
    TARGET = "SalePrice"

    df = cargar_datos(DATA_PATH)

    X, y = preprocesar_datos(df, TARGET)

    X, y = modelo_regresion(X, y)

    modelo_logistico(X, y)

    print("\nEjercicio 2 completado correctamente")

