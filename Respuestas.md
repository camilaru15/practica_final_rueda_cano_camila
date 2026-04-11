# Respuestas — Práctica Final: Análisis y Modelado de Datos

> Rellena cada pregunta con tu respuesta. Cuando se pida un valor numérico, incluye también una breve explicación de lo que significa.

---

## Ejercicio 1 — Análisis Estadístico Descriptivo
---
Añade aqui tu descripción y analisis:
En este ejercicio se ha realizado un análisis estadístico descriptivo del dataset de precios de viviendas.

En primer lugar, se ha explorado la estructura del dataset, identificando el número de variables, tipos de datos y la presencia de valores nulos, observándose que algunas variables presentan un porcentaje elevado de datos faltantes.

Posteriormente, se han calculado estadísticos descriptivos de las variables numéricas, prestando especial atención a la variable objetivo `SalePrice`, que muestra una distribución asimétrica positiva y presencia de valores extremos.

También se han generado visualizaciones como histogramas y matrices de correlación, lo que ha permitido identificar relaciones relevantes entre variables, destacando aquellas con mayor correlación con el precio.

Además, se ha realizado un análisis de variables categóricas y la detección de outliers mediante el método del rango intercuartílico (IQR).

En conjunto, este análisis ha permitido comprender mejor la estructura y características del dataset, sentando las bases para el posterior modelado en los siguientes ejercicios.
---

**Pregunta 1.1** — ¿De qué fuente proviene el dataset y cuál es la variable objetivo (target)? ¿Por qué tiene sentido hacer regresión sobre ella?

>el dataset proviene de la plataforma kaggle,concretamente del conjunto "House Prices – Advanced Regression Techniques", basado en datos reales de viviendas de Ames (Iowa, EE.UU.)

La variable objetivo(target) es `SalePrice`, que representa el precio de venta de las viviendas.

En este dataset tiene sentido aplicar un modelo de regresión porque:
- Se trata de una variable numérica continua.
- El objetivo es predecir su valor a partir de otras características del inmueble(superficie, calidad, ubicación, etc.).
- Existe una relación observable entre múltiples variables explicativas y el precio, lo que hace adecuado el uso de modelos de regresión.

**Pregunta 1.2** — ¿Qué distribución tienen las principales variables numéricas y has encontrado outliers? Indica en qué variables y qué has decidido hacer con ellos.

>La variable objetivo presenta una asimetría positiva (skewness = 1.88), lo que indica que existen valores altos que desplazan la distribución hacia la derecha.

Además, la curtosis es elevada (6.53), lo que sugiere la presencia de colas pesadas y numerosos valores atípicos.

Esto indica que la distribución no es normal y que existen outliers relevantes que pueden afectar al modelado.Juntas te dicen "Los precios no están bien distribuidos y hay valores extremos importantes"

**Pregunta 1.3** — ¿Qué tres variables numéricas tienen mayor correlación (en valor absoluto) con la variable objetivo? Indica los coeficientes.

>Las tres variables numéricas con mayor correlación (en valor absoluto) con la variable objetivo `SalePrice` son:

- OverallQual: 0.790982 (calidad general de la vivienda)
- GrLivArea: 0.708624 (superficie habitable)
- GarageCars: 0.640409 (capacidad del garaje)

Esto indica que el precio de la vivienda está fuertemente influenciado por la calidad de construcción y acabados tiene un impacto directo en el precio, el área habitable y la capacidad del garaje tambien influyen con añadir valor a la propiedad buscando ser coherentes con el mercado inmobiliario

**Pregunta 1.4** — ¿Hay valores nulos en el dataset? ¿Qué porcentaje representan y cómo los has tratado?

> Sí, el dataset presenta valores nulos en varias variables. Algunas de las más relevantes son:

- PoolQC: ~99.5%
- MiscFeature: ~96.3%
- Alley: ~93.7%
- Fence: ~80.7%

Estos valores nulos se han identificado mediante el cálculo del porcentaje de valores faltantes por variable.

En esta fase descriptiva no se han tratado, ya que el objetivo es analizar la estructura del dataset, pero se tendrán en cuenta en el preprocesamiento del modelo en el siguiente ejercicio.

---

## Ejercicio 2 — Inferencia con Scikit-Learn

---
Añade aqui tu descripción y analisis:
En este ejercicio se ha construido un modelo de regresión lineal utilizando Scikit-Learn con el objetivo de predecir el precio de las viviendas (`SalePrice`).

Se ha realizado un preprocesamiento de los datos que incluye la eliminación de variables con muchos valores nulos, la codificación de variables categóricas mediante one-hot encoding y la imputación de valores faltantes.

Posteriormente, se ha dividido el dataset en conjuntos de entrenamiento y test, aplicando escalado a las variables numéricas antes de entrenar el modelo.

Finalmente, se ha evaluado el modelo utilizando métricas como MAE, RMSE y R², y se ha analizado su rendimiento, observando que el modelo presenta un comportamiento aceptable aunque con margen de mejora debido a la presencia de outliers y posibles relaciones no lineales.

---

**Pregunta 2.1** — Indica los valores de MAE, RMSE y R² de la regresión lineal sobre el test set. ¿El modelo funciona bien? ¿Por qué?

> El modelo de regresión lineal ha obtenido las siguientes métricas en el conjunto de test:

- MAE (Mean Absolute Error): 20559.76  
- RMSE (Root Mean Squared Error): 52237.53  
- R² (coeficiente de determinación): 0.6442  

El valor de R² indica que el modelo es capaz de explicar aproximadamente el 64.4% de la variabilidad del precio de las viviendas, lo cual puede considerarse un rendimiento moderado.

El MAE muestra que el error medio de las predicciones es de unos 20,559, lo cual es razonable en el contexto de precios de vivienda. Sin embargo, el RMSE es considerablemente mayor (52,237), lo que indica la presencia de errores grandes en algunas predicciones.

En general, el modelo funciona de forma aceptable, pero no óptima. La diferencia entre MAE y RMSE sugiere que existen outliers o valores extremos que afectan negativamente al rendimiento del modelo.

Esto indica que, aunque el modelo captura una parte importante de la variabilidad del precio, aún existen factores no lineales o variables no consideradas que limitan su capacidad predictiva.

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
