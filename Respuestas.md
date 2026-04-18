# Respuestas — Práctica Final: Análisis y Modelado de Datos

> Rellena cada pregunta con tu respuesta. Cuando se pida un valor numérico, incluye también una breve explicación de lo que significa.

---

## Ejercicio 1 — Análisis Estadístico Descriptivo
---
En este ejercicio se ha realizado un análisis estadístico descriptivo del dataset de precios de viviendas. Se ha comenzado explorando la estructura del dataset, identificando el número de variables, sus tipos y la presencia de valores nulos. Se ha observado que algunas variables presentan un porcentaje elevado de datos faltantes, especialmente aquellas relacionadas con características poco frecuentes. Posteriormente, se han calculado estadísticos descriptivos de las variables numéricas, prestando especial atención a la variable objetivo `SalePrice`. Esta presenta una distribución asimétrica positiva (skewness ≈ 1.88) y alta curtosis (≈ 6.53), lo que indica la presencia de valores extremos y una distribución no normal. Se han generado visualizaciones como histogramas y matrices de correlación, que han permitido identificar relaciones relevantes entre variables. Destacan variables como `OverallQual` (≈ 0.79), `GrLivArea` (≈ 0.70) y `GarageCars` (≈ 0.64) por su alta correlación con el precio. También se ha realizado un análisis de variables categóricas, observando la distribución de sus valores, y se han detectado outliers mediante el método del rango intercuartílico (IQR). En conjunto, este análisis permite comprender mejor la estructura del dataset y proporciona una base sólida para el modelado predictivo posterior
---

**Pregunta 1.1** — ¿De qué fuente proviene el dataset y cuál es la variable objetivo (target)? ¿Por qué tiene sentido hacer regresión sobre ella?

> El dataset proviene de Kaggle y contiene información sobre características de viviendas y sus precios de venta. La variable objetivo es `SalePrice`, que representa el precio de venta de cada vivienda. Tiene sentido aplicar regresión ya que se trata de una variable continua y se busca predecir su valor en función de otras variables explicativas.

**Pregunta 1.2** — ¿Qué distribución tienen las principales variables numéricas y has encontrado outliers? Indica en qué variables y qué has decidido hacer con ellos.

> La variable objetivo presenta una distribución asimétrica positiva, lo que indica que existen valores altos que generan una cola hacia la derecha. Además, la curtosis elevada sugiere la presencia de colas pesadas, lo que implica la existencia de valores atípicos. Estos outliers han sido identificados mediante el método IQR, detectándose un número relevante de ellos. En este ejercicio no se han eliminado, pero se reconoce que pueden afectar al rendimiento de modelos posteriores.

**Pregunta 1.3** — ¿Qué tres variables numéricas tienen mayor correlación (en valor absoluto) con la variable objetivo? Indica los coeficientes.

> Las tres variables con mayor correlación absoluta con `SalePrice` son:
- `OverallQual`: ≈ 0.79
- `GrLivArea`: ≈ 0.70
- `GarageCars`: ≈ 0.64
Estas variables muestran una relación positiva con el precio, indicando que a mayor calidad, superficie habitable o capacidad de garaje, mayor es el valor de la vivienda.

**Pregunta 1.4** — ¿Hay valores nulos en el dataset? ¿Qué porcentaje representan y cómo los has tratado?

> Sí, existen valores nulos en varias variables, algunas con porcentajes muy elevados. En este ejercicio no se han imputado ni eliminado explícitamente, pero se han identificado como un aspecto importante a tratar en fases posteriores de modelado.

---

## Ejercicio 2 — Inferencia con Scikit-Learn

---
En este ejercicio se han desarrollado dos modelos: uno de regresión lineal para predecir el precio de las viviendas y otro de regresión logística para clasificar las viviendas en diferentes rangos de precio. Se ha realizado un preprocesamiento de los datos que incluye la eliminación de variables con alto porcentaje de valores nulos, la codificación de variables categóricas mediante one-hot encoding y la imputación de valores faltantes con la media. Posteriormente, se ha dividido el dataset en conjuntos de entrenamiento y test, aplicando escalado a las variables antes de entrenar los modelos. El modelo de regresión lineal permite estimar el precio de las viviendas, mientras que el modelo de regresión logística permite clasificarlas en categorías (bajo, medio-bajo, medio-alto, alto), proporcionando una perspectiva adicional del problema.
---

**Pregunta 2.1** — Indica los valores de MAE, RMSE y R² de la regresión lineal sobre el test set. ¿El modelo funciona bien? ¿Por qué?

> El modelo de regresión lineal ha obtenido las siguientes métricas:
- MAE: 20559.76
- RMSE: 52237.53
- R²: 0.6442
El modelo explica aproximadamente el 64.4% de la variabilidad del precio, lo que indica un rendimiento moderado. Aunque capta una parte importante de la estructura de los datos, los errores elevados (especialmente reflejados en el RMSE) sugieren que existen observaciones difíciles de predecir, probablemente asociadas a outliers o relaciones no lineales no capturadas por el modelo.

---

## Ejercicio 3 — Regresión Lineal Múltiple en NumPy

---
En este ejercicio se ha implementado un modelo de regresión lineal múltiple desde cero utilizando NumPy, sin recurrir a librerías de machine learning. Se ha aplicado el método de mínimos cuadrados ordinarios (OLS) para estimar los coeficientes del modelo, lo que permite encontrar la mejor combinación lineal de variables que minimiza el error entre valores reales y predichos. Posteriormente, se han evaluado los resultados mediante métricas como MAE, RMSE y R², y se ha comparado el modelo obtenido con los valores teóricos de referencia proporcionados en el enunciado.

---

**Pregunta 3.1** — Explica en tus propias palabras qué hace la fórmula β = (XᵀX)⁻¹ Xᵀy y por qué es necesario añadir una columna de unos a la matriz X.

> La fórmula calcula los coeficientes óptimos del modelo de regresión lineal múltiple mediante el método de mínimos cuadrados. Su objetivo es encontrar los valores de β que minimizan la suma de los errores al cuadrado entre los valores reales y los predichos. Para ello, se utiliza una solución matricial que permite resolver el sistema de ecuaciones de forma eficiente. La matriz X contiene las variables independientes, y la multiplicación XᵀX permite capturar la relación entre ellas. Al invertir esta matriz y multiplicarla por Xᵀy, se obtiene el vector de coeficientes que mejor ajusta el modelo. Se añade una columna de unos a la matriz X para incluir el término independiente (β₀), que representa el valor base de la variable objetivo cuando todas las variables explicativas son cero. Sin este término, el modelo estaría obligado a pasar por el origen, lo cual no es realista en la mayoría de los casos.

**Pregunta 3.2** — Copia aquí los cuatro coeficientes ajustados por tu función y compáralos con los valores de referencia del enunciado.

| Parametro | Valor real | Valor ajustado |
|-----------|-----------|----------------|
| β₀        | 5.0       |      4.864995  |
| β₁        | 2.0       |      2.063618  |
| β₂        | -1.0      |     -1.117038  |
| β₃        | 0.5       |      0.438517  |

> Los coeficientes ajustados son muy cercanos a los valores reales, lo que indica que el modelo ha sido capaz de recuperar correctamente la relación subyacente entre las variables. Las pequeñas diferencias se deben al ruido presente en los datos, lo cual es esperable en un escenario real.

**Pregunta 3.3** — ¿Qué valores de MAE, RMSE y R² has obtenido? ¿Se aproximan a los de referencia?

> Valores obtenidos: 
- MAE: 1.1665 
- RMSE: 1.4612 
- R²: 0.6897 
Los valores obtenidos son coherentes con los valores de referencia del enunciado. El MAE indica un error medio bajo, mientras que el RMSE es ligeramente superior, lo que sugiere la presencia de algunos errores más grandes. El R² cercano a 0.69 indica que el modelo explica una proporción significativa de la variabilidad de los datos, lo que se considera un buen resultado dado que el dataset incluye ruido.

**Pregunta 3.4* — Compara los resultados con la reacción logística anterior para tu dataset y comprueba si el resultado es parecido. Explica qué ha sucedido. 

> En el ejercicio anterior se utilizó un modelo de regresión basado en Scikit-Learn aplicado a un dataset real, mientras que en este ejercicio se ha trabajado con un dataset sintético y un modelo implementado manualmente. Los resultados no son directamente comparables, ya que en este caso el modelo está diseñado para recuperar una relación lineal conocida, mientras que en el ejercicio anterior el modelo debía ajustarse a datos reales con mayor complejidad y ruido. Sin embargo, ambos enfoques muestran un comportamiento coherente: el modelo logra capturar la relación entre variables y obtener métricas razonables, lo que valida tanto la implementación manual como el uso de librerías en contextos más complejos.

---

## Ejercicio 4 — Series Temporales
---
En este ejercicio se ha generado una serie temporal sintética compuesta por cuatro elementos principales: tendencia, estacionalidad, ciclo de largo plazo y ruido gaussiano. Posteriormente, se ha aplicado una descomposición aditiva para separar estos componentes y analizar el comportamiento de cada uno de ellos de forma independiente. Finalmente, se ha evaluado si el residuo se comporta como un ruido blanco ideal mediante estadísticos descriptivos y pruebas estadísticas.
---

**Pregunta 4.1** — ¿La serie presenta tendencia? Descríbela brevemente (tipo, dirección, magnitud aproximada).

>Sí, la serie presenta una tendencia claramente creciente y aproximadamente lineal a lo largo del tiempo. La dirección es positiva, lo que indica que los valores aumentan progresivamente. La magnitud de la tendencia es moderada, ya que el incremento es constante pero no abrupto, consistente con una pendiente suave. Esto es coherente con la forma en que se ha generado la serie, donde se incorpora una tendencia lineal creciente.

**Pregunta 4.2** — ¿Hay estacionalidad? Indica el periodo aproximado en días y la amplitud del patrón estacional.

>Sí, la serie presenta una estacionalidad muy marcada. El periodo es aproximadamente de 365 días, lo que indica un patrón anual. La amplitud del patrón estacional es considerable, con oscilaciones visibles en torno a la tendencia con una amplitud aproximada de ±15 unidades respecto a la tendencia. Esto refleja un comportamiento repetitivo claro a lo largo de los años.

**Pregunta 4.3** — ¿Se aprecian ciclos de largo plazo en la serie? ¿Cómo los diferencias de la tendencia?

>Sí, además de la tendencia y la estacionalidad, se observan ciclos de largo plazo. Estos ciclos tienen una frecuencia mucho menor que la estacionalidad (varios años) y se manifiestan como ondulaciones amplias en la serie. Se diferencian de la tendencia en que no son monotónicos (no crecen de forma continua), sino que presentan fases de subida y bajada. Mientras la tendencia es una evolución sostenida en el tiempo, los ciclos representan fluctuaciones alrededor de esa tendencia.

**Pregunta 4.4** — ¿El residuo se ajusta a un ruido ideal? Indica la media, la desviación típica y el resultado del test de normalidad (p-value) para justificar tu respuesta.

> El residuo se aproxima bastante a un ruido ideal.

Resultados obtenidos:
- Media: 0.1271  
- Desviación típica: 3.2220  
- Asimetría: -0.0509  
- Curtosis: -0.0610  

En cuanto al test de normalidad (Jarque-Bera):
- p-value: 0.576561  

Dado que el p-value es mayor que 0.05, no se rechaza la hipótesis nula de normalidad. Esto indica que el residuo puede considerarse aproximadamente normal.

Además, el test ADF (Augmented Dickey-Fuller) da un p-value de 0.000000, lo que indica que el residuo es estacionario.

En conjunto, el residuo cumple razonablemente con las condiciones de un ruido ideal: media cercana a cero, distribución aproximadamente normal y ausencia de tendencia o estructura sistemática.

---

*Fin del documento de respuestas*
