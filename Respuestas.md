# Respuestas — Práctica Final: Análisis y Modelado de Datos

> Rellena cada pregunta con tu respuesta. Cuando se pida un valor numérico, incluye también una breve explicación de lo que significa.

---

## Ejercicio 1 — Análisis Estadístico Descriptivo
---
En este ejercicio se ha realizado un análisis estadístico descriptivo del dataset de precios de viviendas. Se ha comenzado explorando la estructura del dataset, identificando el número de variables, sus tipos y la presencia de valores nulos. Se ha observado que algunas variables presentan un porcentaje elevado de datos faltantes, especialmente aquellas relacionadas con características poco frecuentes. Posteriormente, se han calculado estadísticos descriptivos de las variables numéricas, prestando especial atención a la variable objetivo `SalePrice`. Esta presenta una distribución asimétrica positiva (skewness ≈ 1.88) y alta curtosis (≈ 6.53), lo que indica la presencia de valores extremos y una distribución no normal. Se han generado visualizaciones como histogramas y matrices de correlación, que han permitido identificar relaciones relevantes entre variables. Destacan variables como `OverallQual`, `GrLivArea` y `GarageCars` por su alta correlación con el precio. También se ha realizado un análisis de variables categóricas, observando la distribución de sus valores, y se han detectado outliers mediante el método del rango intercuartílico (IQR). En conjunto, este análisis permite comprender mejor la estructura del dataset y proporciona una base sólida para el modelado posterior.
---

**Pregunta 1.1** — ¿De qué fuente proviene el dataset y cuál es la variable objetivo (target)? ¿Por qué tiene sentido hacer regresión sobre ella?

> El dataset proviene de Kaggle y contiene información sobre características de viviendas y sus precios de venta. La variable objetivo es `SalePrice`, que representa el precio de venta de cada vivienda. Tiene sentido aplicar regresión ya que se trata de una variable continua y se busca predecir su valor en función de otras variables explicativas.

**Pregunta 1.2** — ¿Qué distribución tienen las principales variables numéricas y has encontrado outliers? Indica en qué variables y qué has decidido hacer con ellos.

> La variable objetivo presenta una distribución asimétrica positiva, lo que indica que existen valores altos que generan una cola hacia la derecha. Además, la curtosis elevada sugiere la presencia de colas pesadas, lo que implica la existencia de valores atípicos. Estos outliers han sido identificados mediante el método IQR, detectándose un número relevante de ellos. En este ejercicio no se han eliminado, pero se reconoce que pueden afectar al rendimiento de modelos posteriores.

**Pregunta 1.3** — ¿Qué tres variables numéricas tienen mayor correlación (en valor absoluto) con la variable objetivo? Indica los coeficientes.

> Las tres variables con mayor correlación absoluta con `SalePrice` son:                    - `OverallQual`: ≈ 0.79                                                                     - `GrLivArea`: ≈ 0.70                                                                        - `GarageCars`: ≈ 0.64                                                                   Estas variables muestran una relación positiva con el precio, indicando que a mayor calidad, superficie habitable o capacidad de garaje, mayor es el valor de la vivienda.

**Pregunta 1.4** — ¿Hay valores nulos en el dataset? ¿Qué porcentaje representan y cómo los has tratado?

> Sí, existen valores nulos en varias variables, algunas con porcentajes muy elevados. En este ejercicio no se han imputado ni eliminado explícitamente, pero se han identificado como un aspecto importante a tratar en fases posteriores de modelado.

---

## Ejercicio 2 — Inferencia con Scikit-Learn

---
En este ejercicio se han desarrollado dos modelos: uno de regresión lineal para predecir el precio de las viviendas y otro de regresión logística para clasificar las viviendas en diferentes rangos de precio. Se ha realizado un preprocesamiento de los datos que incluye la eliminación de variables con alto porcentaje de valores nulos, la codificación de variables categóricas mediante one-hot encoding y la imputación de valores faltantes con la media. Posteriormente, se ha dividido el dataset en conjuntos de entrenamiento y test, aplicando escalado a las variables antes de entrenar los modelos. El modelo de regresión lineal permite estimar el precio de las viviendas, mientras que el modelo de regresión logística permite clasificarlas en categorías (bajo, medio-bajo, medio-alto, alto), proporcionando una perspectiva adicional del problema.
---

**Pregunta 2.1** — Indica los valores de MAE, RMSE y R² de la regresión lineal sobre el test set. ¿El modelo funciona bien? ¿Por qué?

> El modelo de regresión lineal ha obtenido las siguientes métricas:                        - MAE: 20559.76                                                                             - RMSE: 52237.53                                                                            - R²: 0.6442                                                                               El modelo explica aproximadamente el 64.4% de la variabilidad del precio, lo cual representa un rendimiento moderado. El MAE indica un error medio razonable, mientras que el RMSE es significativamente mayor, lo que sugiere la presencia de errores grandes en algunas predicciones, probablemente debido a outliers. En general, el modelo funciona de manera aceptable, aunque con margen de mejora.

---

## Ejercicio 3 — Regresión Lineal Múltiple en NumPy

---
Añade aqui tu descripción y analisis:

---

**Pregunta 3.1** — Explica en tus propias palabras qué hace la fórmula β = (XᵀX)⁻¹ Xᵀy y por qué es necesario añadir una columna de unos a la matriz X.

> _Escribe aquí tu respuesta_

**Pregunta 3.2** — Copia aquí los cuatro coeficientes ajustados por tu función y compáralos con los valores de referencia del enunciado.

| Parametro | Valor real | Valor ajustado |
|-----------|-----------|----------------|
| β₀        | 5.0       |                |
| β₁        | 2.0       |                |
| β₂        | -1.0      |                |
| β₃        | 0.5       |                |

> _Escribe aquí tu respuesta_

**Pregunta 3.3** — ¿Qué valores de MAE, RMSE y R² has obtenido? ¿Se aproximan a los de referencia?

> _Escribe aquí tu respuesta_

**Pregunta 3.4* — Compara los resultados con la reacción logística anterior para tu dataset y comprueba si el resultado es parecido. Explica qué ha sucedido. 

> _Escribe aquí tu respuesta_

---

## Ejercicio 4 — Series Temporales
---
Añade aqui tu descripción y analisis:

---

**Pregunta 4.1** — ¿La serie presenta tendencia? Descríbela brevemente (tipo, dirección, magnitud aproximada).

> _Escribe aquí tu respuesta_

**Pregunta 4.2** — ¿Hay estacionalidad? Indica el periodo aproximado en días y la amplitud del patrón estacional.

> _Escribe aquí tu respuesta_

**Pregunta 4.3** — ¿Se aprecian ciclos de largo plazo en la serie? ¿Cómo los diferencias de la tendencia?

> _Escribe aquí tu respuesta_

**Pregunta 4.4** — ¿El residuo se ajusta a un ruido ideal? Indica la media, la desviación típica y el resultado del test de normalidad (p-value) para justificar tu respuesta.

> _Escribe aquí tu respuesta_

---

*Fin del documento de respuestas*
