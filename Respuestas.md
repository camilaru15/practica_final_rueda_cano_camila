# Respuestas — Práctica Final: Análisis y Modelado de Datos

> Rellena cada pregunta con tu respuesta. Cuando se pida un valor numérico, incluye también una breve explicación de lo que significa.

---

## Ejercicio 1 — Análisis Estadístico Descriptivo
---
Añade aqui tu descripción y analisis:

---

**Pregunta 1.1** — ¿De qué fuente proviene el dataset y cuál es la variable objetivo (target)? ¿Por qué tiene sentido hacer regresión sobre ella?

>el dataset proviene de la plataforma kaggle,concretamente del conjunto "House Prices – Advanced Regression Techniques", basado en datos reales de viviendas de Ames (Iowa, EE.UU.)https://www.kaggle.com/datasets/rishitaverma02/house-prices-advanced-regression-techniques?select=train+%281%29.csv

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

---

**Pregunta 2.1** — Indica los valores de MAE, RMSE y R² de la regresión lineal sobre el test set. ¿El modelo funciona bien? ¿Por qué?

> _Escribe aquí tu respuesta_


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
