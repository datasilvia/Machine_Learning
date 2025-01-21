# Machine Learning

# 1. ¿Qué es Machine Learning?

# 2. Primeros pasos para entrenar un modelo de Machine Learning

1- Importar las librerias necesarias

``` python
import pandas as pd
```

2- Importar los datos que vayamos a usar

``` python
df = pd.read_csv(dataset.csv)
```

3- Examinar los datos

``` python
df.head()
df.shape
df.columns
df.types
```

4- Limpiar los datos

5- Definir Target y Features

```python
target = df['column']

features = df.drop(columns = ['column'])
```

6- Dividir los datos en Training data y Test data (función train_test_split)

``` python

from sklearn.model_selection import train_test_split

X = features

y = target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

```

# 3. Identificar el modelo que vamos a usar. 

Esto dependerá del tipo de problema que vayamos a tratar, porque usaremos un algoritmo u otro.

¿Qué tipos de problemas hay?

- Clasificación
    - Regresión Logística
    - Naive Bayes

- Regresión 
    - KNN

    Este algoritmo se basa en los vecinos más próximos, en distancias. Por tanto requiere **Variables numéricas**.
    


## Clasificación

### Algoritmo de Naive Bayes

Puedes aprender más [aquí](https://www.ibm.com/es-es/topics/naive-bayes)

Se utiliza para tareas de clasificación como la clasificación de textos. 
Utiliza principios de probabilidad para realizar tareas de clasificación. 
Naïve Bayes forma parte de una familia de algoritmos de aprendizaje generativo, 
lo que significa que busca modelar la distribución de las entradas de una clase o categoría determinada. 
A diferencia de los clasificadores discriminativos, como la regresión logística, 
no aprende qué características son las más importantes para diferenciar entre clases.


## Regresión

``` python
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
 ```

### KNN

Puedes aprender más [aquí](https://www.ibm.com/es-es/topics/knn)


``` python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
```






# 4. Entrenar el modelo

``` python

model.fit(X_train, y_train)

```


# 5. Probar el modelo

``` python

y_pred = model.predict(X_test)

```


# 6. Evaluar el modelo

Las métricas para la evaluación del desempeño del modelo serán diferentes dependiendo si estamos tratando con un problema de clasificación o de regresión.

## Métricas para Clasificación:

    - Accuracy (Precisión global)

    - Recall

    - Precision

    - F1 Score


## Métricas para Regresión:

    - R2 (Mide cómo de efectivamente las variables independientes en un modelo de regresión predicen o explican los cambios en la variable dependiente). Varía entre 0 (pésimo) y 1 (Eexcelente)



# Evaluar el modelo (código)

``` python

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, precision_recall_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Hacer predicciones
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # Probabilidad de la clase positiva

# 1. Reporte de clasificación
print("Reporte de Clasificación:")
print(classification_report(y_test, y_pred))

# 2. Matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["No Fraude", "Fraude"], yticklabels=["No Fraude", "Fraude"])
plt.title("Matriz de Confusión")
plt.ylabel("Etiqueta Real")
plt.xlabel("Etiqueta Predicha")
plt.show()

# 3. Métrica ROC-AUC
roc_auc = roc_auc_score(y_test, y_prob)
print(f"ROC-AUC: {roc_auc:.2f}")

# 4. Curva Precision-Recall
precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
pr_auc = auc(recall, precision)

plt.figure(figsize=(6, 4))
plt.plot(recall, precision, marker='.', label=f"PR AUC = {pr_auc:.2f}")
plt.title("Curva Precision-Recall")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.show()


```

## Evaluación de un modelo de Regresión (código)

``` python

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Hacer predicciones con el modelo
y_pred = knn.predict(X_test)

# 1. Calcular métricas de evaluación
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)  # Raíz cuadrada del MSE
r2 = r2_score(y_test, y_pred)

print("Métricas de Evaluación:")
print(f"MAE  (Mean Absolute Error): {mae:.2f}")
print(f"MSE  (Mean Squared Error): {mse:.2f}")
print(f"RMSE (Root Mean Squared Error): {rmse:.2f}")
print(f"R^2  (Coeficiente de determinación): {r2:.2f}")

# 2. Visualizar resultados: Gráfico de dispersión
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label="Línea ideal")
plt.title("Comparación entre valores reales y predichos")
plt.xlabel("Valores Reales (y_test)")
plt.ylabel("Valores Predichos (y_pred)")
plt.legend()
plt.grid()
plt.show()

# 3. Histograma del error
errors = y_test - y_pred
plt.figure(figsize=(6, 4))
plt.hist(errors, bins=30, alpha=0.7, color="blue", edgecolor="black")
plt.title("Distribución del Error")
plt.xlabel("Error (y_test - y_pred)")
plt.ylabel("Frecuencia")
plt.grid()
plt.show()

```



-----------------------------------------------------------------------------------------------

# Feature engineering (Ingeniería de clases)

Se utiliza para mejorar el desempeño (performance) del modelo y reducir la complejidad.


Como cada modelo requiere unos tipos de datos específicos, a veces tenemos que **transformar los datos**.

Hay varios métodos para ello:

# One hot encoding

Consiste en transformar una variable categórica en una serie de variables numéricas.

**Ventaja**: El performance es mejor, ya que podemos aplicar cálculos matriciales.

**Desventaja**: Aumenta la dimensionalidad de manera significativa.


# Label encoding

Consiste en asignar valores numéricos a valores categóricos.

**Ventajas**: 

- Se mantiene la dimensionalidad.

- Es fácil volver a la situación original para interpretar los datos.

**Desventajas**:

- Puede llegar a introducir una relación ordinal falsa entre las categorías (cuando no existe tal relación, por ejemplo, los colores, ya que no se dividen de modo jerárquico).

Para evitar esta relñación ordinal falsa, debemos recurrir al One hot encoding.


# Binary encoding

Es una mezcla entre el **one hot encoding** y el **label encoding**.

Se usa para **mejorar la performance** de los modelos.

Se utiliza cuando exista un dataset muy grande.


# Binning

Consiste en agrupar las variables contínuas en intervalos de variables categóricas. (Por ejemplo, agrupar edades en "jóven", "adulto", "anciano").

Su **ventaja** es que **reduce la complejidad** del modelo.


# Feature Scaling

Consiste en transformar las características numéricas para que estén en escalas similares. Se puede usar normalización o min-max scaler

### MinMaxScaler 

Trabajamos con el rango mínimo-máximo y lo transformamos en un rango 0 a 1.

Es una **normalización** que escala las características en un **rango de 0 a 1**, manteniendo la relación entre los datos.

Este método **mejora la performance** de los modelos.


### Z-Score (Standard scaling)

Consiste en llevar la media a 0 y el desvío standard a 1.

Se usa cuando tenemos característiocas que tienen un comportamiento normal.

Es una **estandarización**.


### Robust Scaling

Es similar el Standard scaling (Z-Score), pero en vez de utilizar el desvío estandar, utilizamos el rango intercuartílico (IQR), y en vez de utilizar la media, utilizamos la mediana.

Esto perfomará mejor cuando los datos **NO** provengan de una distribución normal.



# PCA (Análisis de componentes principales)

Se trata de reducir la dimensionalidad de un dataset, y quedarse con las variables que explican la correlación entre las variables.

-------------------------------------------------------------------------------------------------

# Feature Selection

Se usa para mejorar el rendimiento, y consiste en analizar la relación y eliminar las variables que tengan la relación más baja.

- Buscamos features **altamente relacionadas con el target**, pero NO entre ellas.

- Una alta correlación entre features puede causar redundancia e inestabilidad en los modelos,, y puede degradar la performance.



Cuando tengamos **variables categóricas** usaremos chi2

Para **variables numéricas** usaremos la matriz de correlación (mapa de calor).


Existen varios métodos para la feature selection:

-Métodos basados en filtros: 

    -Pruebas de chi2

    -Correlación de Pearson

-Uso de Wrappers

-Uso de embeddings

(Forward selection, backward elimination, recursive, features elimination, ...)




--------------------------------------------------------------------------------------------------

## Métricas adecuadas para datasets desbalanceados:

- **F1-Score**: Es el promedio armónico entre la precisión (precision) y la exhaustividad (recall). Es útil cuando las clases minoritarias son importantes.
- **Precision**: Evalúa qué proporción de las predicciones positivas son correctas.
- **Recall (Sensibilidad)**: Evalúa qué proporción de las muestras positivas reales fueron correctamente identificadas.
- **ROC-AUC (Área bajo la curva ROC)**: Mide la capacidad del modelo para diferenciar entre clases.
- **PR-AUC (Área bajo la curva Precision-Recall)**: Específicamente útil para datasets desbalanceados porque se centra en las clases positivas.

(El Accuracy **NO** es una métrica adecuada para datasets desbalanceados)


------------------------------------------------------------------------------------------------
- Random Forest

- Decision Tree

- SVM

- Naive-Bayes









If you want to read a Jupyter Notebook (files with the .ipynb extension) without installing anything, you can use: https://nbviewer.org/ 
If you also want to edit it, then you can use: https://notebooks.gesis.org/binder/ 