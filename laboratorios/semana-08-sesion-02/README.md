
# Semana 8, Sesión 2: Exploración y visualización de datos

> Objetivo de la sesión: explorar el dataset de nuestro proyecto con estadística descriptiva y visualizaciones para identificar patrones, valores atípicos y problemas que podrían afectar un modelo antes de entrenarlo.

**Lectura previa:** [A Data Scientist's Essential Guide to Exploratory Data Analysis](https://towardsdatascience.com/a-data-scientists-essential-guide-to-exploratory-data-analysis-25637eee0cf6/)

**Referencia:** [Machine Learning Zoomcamp, Módulo 1](https://github.com/DataTalksClub/machine-learning-zoomcamp)

## Requisitos

Para esta sesión utilizamos Python, pandas y matplotlib.

Las dependencias están en `requirements.txt` y los ejercicios se pueden ejecutar utilizando Docker.

Los gráficos generados se guardan en las carpetas `figuras/` correspondientes a cada ejercicio.
pandas y matplotlib ya vienen en el Codespace. En tu laptop, instálalos con `pip install -r requirements.txt` desde la raíz del repositorio. Verifica con:

```
python3 -c "import matplotlib; print(matplotlib.__version__)"
```

Los scripts de hoy guardan cada gráfico como imagen en una carpeta `figuras/`. Ábrelas desde el explorador de archivos de VS Code. También puedes correr el mismo código en un notebook de Jupyter (en el Codespace, crea un archivo `.ipynb` y elige el kernel de Python 3.11), donde el gráfico aparece debajo de la celda.

## Antes de empezar

### ¿Qué preguntas responde la exploración de datos antes de entrenar cualquier modelo?

La exploración de datos permite conocer cómo están distribuidos los datos, identificar valores faltantes, duplicados y errores, además de encontrar relaciones entre variables. Esto ayuda a decidir qué datos se deben limpiar o transformar antes de entrenar un modelo.

### ¿Qué herramientas se utilizan para explorar una variable y la relación entre dos variables?

Para analizar una sola variable podemos utilizar estadísticas descriptivas, histogramas y diagramas de caja.

Para analizar la relación entre dos variables podemos utilizar gráficos de dispersión, comparaciones entre categorías y correlaciones cuando trabajamos con variables numéricas.

## Pregunta de apertura

**Entrenas un modelo que predice la tarifa de un viaje en taxi. Sin que lo sepas, 15 de los 3000 viajes tienen una tarifa de 999.99 por un error de captura. ¿Qué le pasa al modelo? ¿Lo notarías mirando solo su resultado final?**

Estos valores incorrectos pueden afectar el promedio de las tarifas y hacer que el modelo aprenda patrones que no representan la realidad. Si únicamente revisamos el resultado final, podríamos no identificar el problema.

Por eso es importante explorar los datos antes de entrenar el modelo, utilizando estadísticas descriptivas y gráficos que permitan detectar valores sospechosos.

## Ejercicio 1: Resumen, distribución y atípicos

Dataset: `starter/viajes_taxi.csv`
Archivo: [`starter/ejercicio1_explorar.py`](./starter/ejercicio1_explorar.py), sobre [`starter/viajes_taxi.csv`](./starter/viajes_taxi.csv): 3000 viajes en taxi de Nueva York, con fecha y hora de inicio, zona de origen y de destino, distancia, duración, pasajeros, método de pago, tarifa y propina. El script ya lee el archivo, arma los gráficos y los guarda en `figuras/`; faltan 4 `TODO`. Córrelo una vez antes de tocar nada:

```
cd laboratorios/semana-08-sesion-02/starter
python3 ejercicio1_explorar.py
```

**Respuesta 1 (TODO 1), antes de ver la solución:** en `describe()`, ¿qué dos valores de la tabla te harían sospechar de un error sin necesidad de graficar nada?

El archivo contiene 3000 registros de viajes en taxi de Nueva York, incluyendo distancia, duración, número de pasajeros, método de pago, tarifa y propina.

### Respuesta 1: ¿Qué dos valores de describe() te harían sospechar de un error?
**Respuesta 2 (TODO 2):** abre `figuras/hist_tarifa.png` después del TODO 2. ¿Por qué casi todo el histograma queda comprimido a la izquierda?

Los dos valores que llaman la atención son:

- La duración mínima de un viaje es -74 minutos, lo cual es imposible.
- La tarifa máxima es 999.99 dólares, un valor demasiado alto comparado con la mediana de 22.60 dólares.

Estos resultados indican que existen registros que debemos revisar antes de utilizar los datos para entrenar un modelo.

### Respuesta 2: La regla IQR marca más de 300 viajes con tarifa atípica. ¿Todos son errores?

No necesariamente. La regla IQR identifica valores que están fuera del rango habitual, pero eso no significa que sean incorrectos.

Por ejemplo, algunos viajes tienen una tarifa exacta de 70 dólares, que podría corresponder a una tarifa fija o una condición especial.

Para determinar si son errores, revisaría la distancia, duración, origen, destino y las reglas tarifarias de esos viajes.

### Implementación

El código se encuentra en `starter/EDA.py`.

Se realizaron las siguientes operaciones:

1. Cargar el dataset.
2. Revisar los tipos de datos.
3. Identificar valores nulos y duplicados.
4. Obtener estadísticas descriptivas.
5. Identificar valores atípicos mediante IQR.
6. Generar gráficos de distribución.
7. Calcular las correlaciones entre variables numéricas.

También se corrigió un error del código original.

La instrucción:

`print(data.corr())`

No funcionaba porque el dataset contiene columnas de texto.

Se reemplazó por:

```python
print(data.corr(numeric_only=True).round(3))
```

De esta manera se calculan las correlaciones utilizando únicamente columnas numéricas.

### Resultados

En el dataset original, la correlación entre distancia y tarifa es 0.229.

Después de eliminar los valores atípicos mediante IQR, esta correlación aumenta a 0.971.

Esto demuestra que los valores extremos pueden afectar considerablemente las relaciones estadísticas entre las variables.

Sin embargo, no significa que todos los valores atípicos deban eliminarse automáticamente.

## Durante el ConcepTest
Esta actividad no se realizó durante la sesión, por lo que no se registró un voto ni una discusión en pareja.

### Tu voto antes de discutir en pareja

Pendiente de completar con la opción seleccionada durante la actividad de clase.

### Tu razonamiento

Si queremos predecir la tarifa de un taxi cuando el pasajero solicita el viaje, debemos utilizar únicamente información disponible en ese momento.

Aunque la duración real tenga una correlación alta con la tarifa, todavía no conocemos ese valor cuando se solicita el taxi.

### ¿Cambiaste de voto después de discutir en pareja? ¿Por qué?

Pendiente de completar según lo realizado durante la actividad de clase.

## Ejercicio 2: Relaciones entre variables
Solo o en pareja. Archivo: [`starter/ejercicio2_relaciones.py`](./starter/ejercicio2_relaciones.py), sobre el mismo `viajes_taxi.csv`. Quita solo los errores que el Ejercicio 1 ya confirmó y completa los 5 `TODO`: dispersión de distancia contra tarifa, propina por método de pago, correlaciones con la tarifa, y pasajeros vacíos o en 0. Al final, el script imprime una comprobación por TODO: completa los 5 hasta que todas digan `OK`.

```
python3 ejercicio2_relaciones.py
```

El código se encuentra en:

`plantilla-ejercicio2/explorar_relaciones.py`

Se utilizó el mismo dataset del ejercicio 1, eliminando únicamente los registros con errores identificados, como tarifas de 999.99 dólares o duraciones negativas.

Después de limpiar los datos, quedaron 2975 viajes de los 3000 originales.

### Hallazgo 1: Relación entre distancia y tarifa

El gráfico de dispersión muestra una relación positiva entre la distancia recorrida y la tarifa del viaje.

En general, mientras mayor sea la distancia, mayor será el costo.

La correlación entre ambas variables es de 0.960, lo que indica una relación lineal positiva bastante fuerte.

También encontramos 300 viajes con una tarifa exacta de 70 dólares. Estos registros forman una línea horizontal en el gráfico y podrían representar tarifas especiales.

Por esta razón, no deberían eliminarse sin antes investigar su origen.

Gráfico: `plantilla-ejercicio2/figuras/distancia_vs_tarifa.png`

### Hallazgo 2: Propina según el método de pago

Al analizar las propinas según el método de pago encontramos:

- 888 viajes pagados en efectivo, todos con propina registrada de 0 dólares.
- 2027 viajes pagados con tarjeta, con una propina promedio de 4.73 dólares.
- 60 registros sin información del método de pago.

Esto no significa necesariamente que los pasajeros que pagan en efectivo nunca dejen propina.

Es posible que las propinas en efectivo no se registren de la misma manera que las realizadas con tarjeta.

Por lo tanto, este resultado podría estar relacionado con la forma en que se almacenan los datos.

### Hallazgo 3: ¿Se puede utilizar la duración del viaje para predecir la tarifa?

La duración del viaje tiene una correlación de 0.956 con la tarifa, lo que demuestra una relación positiva fuerte.

Sin embargo, no utilizaría la duración real si queremos predecir la tarifa al momento de solicitar el taxi.

Esto se debe a que todavía no conocemos cuánto durará realmente el viaje.

Utilizar esa información provocaría fuga de información (*data leakage*), porque estaríamos utilizando datos que solo estarán disponibles después de completar el viaje.

En su lugar, podríamos utilizar una duración estimada según la distancia, la ruta y las condiciones del tráfico.

### Otros resultados

También encontramos:

- 119 viajes con el número de pasajeros vacío.
- 19 viajes con cero pasajeros.
- Una correlación de 0.633 entre propina y tarifa.

Estos resultados muestran que existen datos que deberían revisarse antes de entrenar un modelo.

## Ejercicio 3: EDA de nuestro proyecto de Transfermarkt

### Descripción del proyecto

Para este ejercicio utilizamos el dataset de nuestro proyecto de Ciencia de Datos, que contiene información de jugadores de fútbol obtenida de Transfermarkt.

El archivo utilizado es:

`datos-proyecto/players.csv.gz`

Nuestro objetivo es analizar el valor de mercado de los jugadores y estudiar cómo se relaciona con características como la edad y la posición.

La variable objetivo es `market_value_in_eur`.

El análisis se realizó utilizando Python, pandas y matplotlib.

### Paso 1: Resumen del dataset

El dataset contiene 50.149 jugadores y 26 columnas.

Al revisar los datos encontramos:

- 41.528 jugadores con valor de mercado registrado y positivo.
- 8.621 jugadores sin valor de mercado.
- 49 registros sin fecha de nacimiento.
- 4.365 registros sin información de estatura.
- No se encontraron identificadores `player_id` duplicados.

Los valores faltantes son importantes porque pueden afectar el entrenamiento del modelo.

Especialmente, los jugadores sin valor de mercado no podrían utilizarse directamente para entrenar un modelo supervisado que intente predecir esa variable.

### Distribución de la variable objetivo

Al analizar el valor de mercado encontramos:

- Mediana: 275.000 euros.
- Promedio: aproximadamente 1.511.049 euros.
- Valor máximo: 200.000.000 euros.

La diferencia entre el promedio y la mediana demuestra que la distribución está bastante concentrada en valores bajos, pero existen jugadores con valores de mercado muy elevados.

Estos jugadores hacen que el promedio sea mucho mayor que la mediana.

Por esta razón, la mediana puede representar mejor el valor típico de los jugadores del dataset.

Gráfico: `plantilla-eda/figuras/distribucion_objetivo.png`

### Paso 2: Relaciones entre variables

#### Relación entre edad y valor de mercado

Para analizar la relación entre edad y valor de mercado, se calculó una edad aproximada utilizando como referencia el 1 de julio del año de la última temporada registrada.

Se consideraron edades entre 15 y 45 años, quedando 41.492 jugadores para este análisis.

La correlación de Pearson entre edad y valor de mercado fue de -0.040.

Esto indica que no existe una relación lineal fuerte entre ambas variables considerando todos los jugadores.

Sin embargo, esto no significa que la edad no influya en el valor de mercado.

Es posible que la relación no sea lineal, ya que algunos jugadores aumentan su valor durante sus primeros años profesionales y después lo disminuyen cuando se acercan al final de su carrera.

También pueden existir diferencias según la posición y otras características.

Gráfico: `plantilla-eda/figuras/relacion_objetivo.png`

#### Relación entre posición y valor de mercado

Al comparar los valores de mercado según la posición encontramos las siguientes medianas:

| Posición | Valor mediano |
|---|---:|
| Ataque | €300.000 |
| Defensa | €300.000 |
| Mediocampo | €300.000 |
| Portero | €150.000 |

Los porteros tienen un valor de mercado mediano menor que los jugadores de las otras posiciones principales.

Esto sugiere que la posición podría ser una variable útil para nuestro modelo predictivo.

### Paso 3: Tres hallazgos importantes para el modelado

#### Hallazgo 1: Valores faltantes

Encontramos 8.621 jugadores sin un valor de mercado registrado.

Esto representa un problema porque el valor de mercado es la variable que queremos predecir.

No sería correcto reemplazar automáticamente esos valores con el promedio, ya que estaríamos inventando información para entrenar el modelo.

Por eso, para un entrenamiento supervisado utilizaríamos inicialmente los registros que sí tienen un valor de mercado válido.

También deberíamos revisar los registros sin estatura o fecha de nacimiento, dependiendo de las variables que decidamos utilizar.

#### Hallazgo 2: Valores atípicos

Utilizando el método IQR encontramos que el límite superior para identificar valores de mercado atípicos es de 1.600.000 euros.

En total, 5.799 jugadores superan ese límite, lo que representa aproximadamente el 13.96 % de los jugadores con valor de mercado registrado.

Sin embargo, estos valores no necesariamente son errores.

En el fútbol existen jugadores con valores de mercado muy superiores al promedio debido a su rendimiento, edad, trayectoria y otros factores.

Por esa razón, no eliminaríamos automáticamente esos registros.

Una alternativa sería probar una transformación logarítmica de la variable objetivo para reducir la influencia de los valores extremos.

#### Hallazgo 3: Relación entre las características y el valor de mercado

La correlación entre la edad y el valor de mercado fue de -0.040, indicando que no existe una relación lineal fuerte en el conjunto completo de jugadores.

Por otro lado, encontramos diferencias en el valor mediano según la posición, especialmente en el caso de los porteros.

Esto nos indica que no deberíamos depender únicamente de la edad para predecir el valor de un jugador.

Sería conveniente incluir otras características, como la posición, y probar modelos que puedan identificar relaciones no lineales.

### Limitaciones del análisis

Una limitación importante es que el dataset contiene información de jugadores de diferentes temporadas.

Además, el valor de mercado y las características del jugador no necesariamente corresponden a una misma fecha.

Por eso, la edad utilizada en este análisis es aproximada.

Antes de entrenar un modelo, sería importante verificar que los datos utilizados para predecir el valor de mercado correspondan al momento adecuado y no incluyan información futura.

### Conclusión del ejercicio 3

El análisis exploratorio nos permitió conocer mejor los datos de nuestro proyecto e identificar problemas que debemos considerar antes de entrenar el modelo.

Encontramos valores faltantes, una distribución muy desigual del valor de mercado y diferencias entre las posiciones de los jugadores.

También observamos que la edad por sí sola no presenta una relación lineal fuerte con el valor de mercado.

Por esta razón, sería conveniente utilizar varias características y evaluar diferentes métodos de modelado.

En general, el EDA nos ayuda a comprender mejor los datos y tomar decisiones sobre la limpieza, transformación y selección de variables.

## Antes de la próxima sesión

**Lectura:** [Feature Engineering that Makes Business Sense](https://towardsdatascience.com/feature-engineering-that-makes-business-sense-419b9403236e/)

### La tarifa fija de JFK y la hora del día no existen como columnas del dataset de taxis. ¿Cómo le darías esa información a un modelo?

Crearía nuevas columnas a partir de la información que ya tenemos.

Por ejemplo, una columna que indique si el viaje corresponde a una ruta con tarifa fija hacia o desde JFK, utilizando el origen, destino y las reglas tarifarias correspondientes.

También podría crear una columna con la hora del día a partir de `fecha_hora_inicio`, e incluso identificar si el viaje ocurre en la mañana, tarde o noche.

De esta manera, el modelo tendría información adicional para identificar patrones en las tarifas según la ruta y el horario.

## Ejecución de los ejercicios con Docker

Desde la carpeta raíz del laboratorio, utilizamos los siguientes comandos.

### Ejercicio 1

```powershell
docker compose up --build
```

### Ejercicio 2

```powershell
docker compose run --rm python-app python plantilla-ejercicio2/explorar_relaciones.py
```

### Ejercicio 3

```powershell
docker compose run --rm python-app python plantilla-eda/explorar_proyecto.py
```

Los gráficos generados se guardan en las carpetas correspondientes de cada ejercicio.

## Conclusión general

Durante esta sesión aprendimos a utilizar estadísticas descriptivas y visualizaciones para entender mejor un dataset antes de entrenar un modelo.

Los ejercicios de taxis mostraron cómo los valores erróneos pueden alterar los resultados y cómo una correlación alta no significa necesariamente que una variable pueda utilizarse para predecir.

En nuestro proyecto de Transfermarkt encontramos valores faltantes, valores extremos y características que podrían ayudarnos a predecir el valor de mercado de los jugadores.

La principal conclusión es que realizar un análisis exploratorio antes de entrenar un modelo permite identificar problemas, comprender las relaciones entre variables y tomar mejores decisiones sobre cómo preparar los datos.
