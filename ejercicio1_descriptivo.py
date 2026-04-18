"""
=============================================================================
PRÁCTICA FINAL — EJERCICIO 1
Análisis Estadístico Descriptivo
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Crear carpeta output si no existe
os.makedirs("output", exist_ok=True)


# =============================================================================
# CARGA DE DATOS
# =============================================================================

def cargar_datos(ruta):
    """
    Carga el dataset desde un fichero CSV
    """
    return pd.read_csv(ruta)


# =============================================================================
# RESUMEN ESTRUCTURAL
# =============================================================================

def resumen_estructural(df):
    """
    Muestra información general del dataset
    """
    print("\n--- RESUMEN ESTRUCTURAL ---")
    print(f"Filas: {df.shape[0]}")
    print(f"Columnas: {df.shape[1]}")

    print("\nTipos de datos:")
    print(df.dtypes)

    print("\nValores nulos (%):")
    print((df.isnull().mean() * 100).sort_values(ascending=False))


# =============================================================================
# ESTADÍSTICOS DESCRIPTIVOS
# =============================================================================

def estadisticos_descriptivos(df, target):
    """
    Calcula estadísticas principales del dataset
    """
    df.describe().to_csv("output/ej1_descriptivo.csv")

    Q1 = df[target].quantile(0.25)
    Q3 = df[target].quantile(0.75)
    IQR = Q3 - Q1

    print("\n--- ESTADÍSTICOS TARGET ---")
    print(f"IQR: {IQR}")
    print(f"Skewness: {df[target].skew()}")
    print(f"Curtosis: {df[target].kurtosis()}")

# =============================================================================
# VISUALIZACIONES
# =============================================================================

def graficar_histogramas(df):
    """
    Histogramas de variables numéricas
    """
    df.select_dtypes(include=np.number).hist(figsize=(15, 10))
    plt.tight_layout()
    plt.savefig("output/ej1_histogramas.png")
    plt.close()


def graficar_heatmap(df):
    """
    Matriz de correlación
    """
    corr = df.corr(numeric_only=True)

    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Matriz de correlación")

    plt.savefig("output/ej1_heatmap_correlacion.png")
    plt.close()

# =============================================================================
# VARIABLES CATEGÓRICAS
# =============================================================================

def analizar_categoricas(df):
    """
    Análisis de variables categóricas
    """
    cat_cols = df.select_dtypes(include=["object", "string"]).columns

    for col in cat_cols:
        print(f"\n--- {col} ---")
        print(df[col].value_counts(normalize=True))

# =============================================================================
# BOXPLOTS Y OUTLIERS
# =============================================================================

def boxplots_por_categoria(df, target):
    """
    Boxplots del target según variables categóricas
    """
    cat_cols = df.select_dtypes(include=["object", "string"]).columns[:3]

    plt.figure(figsize=(15, 5))

    for i, col in enumerate(cat_cols):
        plt.subplot(1, 3, i + 1)
        sns.boxplot(x=col, y=target, data=df)
        plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig("output/ej1_boxplots.png")
    plt.close()


def detectar_outliers(df, target):
    """
    Detección de outliers con IQR
    """
    Q1 = df[target].quantile(0.25)
    Q3 = df[target].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[(df[target] < lower) | (df[target] > upper)]

    with open("output/ej1_outliers.txt", "w") as f:
        f.write(f"Outliers detectados: {len(outliers)}\n")

    print(f"\nOutliers detectados: {len(outliers)}")

# =============================================================================
# CORRELACIONES
# =============================================================================

def analizar_correlaciones(df, target):
    """
    Correlaciones con la variable objetivo
    """
    corr = df.corr(numeric_only=True)[target].abs().sort_values(ascending=False)

    print("\n--- TOP CORRELACIONES ---")
    print(corr[1:4])

    print("\n--- MULTICOLINEALIDAD (|r| > 0.9) ---")
    matriz = df.corr(numeric_only=True).abs()

    for i in range(len(matriz.columns)):
        for j in range(i):
            if matriz.iloc[i, j] > 0.9:
                print(matriz.columns[i], "-", matriz.columns[j], matriz.iloc[i, j])

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    DATA_PATH = "data/dataset.csv"
    TARGET = "SalePrice"

    df = cargar_datos(DATA_PATH)

    resumen_estructural(df)

    estadisticos_descriptivos(df, TARGET)

    graficar_histogramas(df)

    graficar_heatmap(df)

    analizar_categoricas(df)

    boxplots_por_categoria(df, TARGET)

    detectar_outliers(df, TARGET)

    analizar_correlaciones(df, TARGET)

    print("\nEjercicio 1 completado correctamente")
