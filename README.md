# MachineLearning_for_Trading

Proyecto en el que estudio la implementación de un sistema inteligente capaz de realizar inversiones en el mercado de FOREX.

La idea básica del proyecto se puede dividir en varias fases:

- Diseño de un sistema predictivo del precio
- Diseño de un agente inteligente que realice la operativa de entradas en corto-largo en un par forex dado (ej. eur-usd)
- Diseño de una interfaz con el terminal del broker para obtener el data-feed en tiempo real y el estado de la cuenta / operaciones.

Como guía de desarrollo, estoy utilizando como lenguaje PYTHON y estas librerías más representativas:

- scikit-learn : utilidades ML varias (cross-validation, GridSearch, LinearRegressores, ...)
- tensorflow   : librería deep learning de bajo nivel
- keras        : front-end de tensorflow (nivel de aplicación)
- featurestools: feature engineering
- deap		   : implementación de algoritmos genéticos

De momento estoy centrado en la implementación de la 1ª fase (red predictiva) para la que he diseñado una red LSTM de varias capas a la que alimento con los precios OHLC, varios indicadores técnicos y detectores de patrones de velas únicamente.

Para la siguiente fase, la idea es utilizar una red A3C (aprendizaje por refuerzo) para que utilice la predicción de la red LSTM junto con el estado de la cuenta en el broker y los estados de las operaciones en curso, para decidir qué hacer (mantener la posición, abrir en corto o en largo).
  
## Changelog

----------------------------------------------------------------------------------------------
##### 05.12.2018 ->commit:"Diseñando gym-environment"
- [x] Inicio el desarrollo del Gym-env que me sirva de referencia para evaluar al agente A3C.
- [x] Clono varios repos en ./gym 

----------------------------------------------------------------------------------------------
##### 04.12.2018 ->commit:"Iniciando diseño A3C-LSTM"
- [x] Descarto filtro post-predictor ya que no introduce ninguna mejora.
- [x] Creo notebook para generar dataframe de predicciones y generar CSV para el Agente A3C
- [x] Inicio el notebook de implementación del agente A3C-LSTM
- [x] Creo RAR files para archivos CSV pesados y extraigo los CSV de la copia de backup

----------------------------------------------------------------------------------------------
##### 03.12.2018 ->commit:"Descarto RFR y reorganizo directorios"
- [x] Descarto el RFR, reorganizo la estructura interna de los directorios y continúo con el filtro post-predictor.

----------------------------------------------------------------------------------------------
##### 30.11.2018 ->commit:"Implementación del RandomForestRegressor"
- [x] Implemenento el RFC y estoy a la espera de que termine el 'fitting' para luego evaluarlo. Lleva ya más de 30 horas...

----------------------------------------------------------------------------------------------
##### 28.11.2018 ->commit:"Realización de estudios de mejora LSTM"

- [x] Incluyo diversos estudios (notebooks) para mejorar el rendimiento de la red predictiva LSTM. Ver notebooks "Study X...."

----------------------------------------------------------------------------------------------
##### 27.11.2018 ->commit:"Terminado desarrollo LSTM"

- [x] Finalizo el desarrollo de la red LSTM que predecirá los precios HIGH-LOW de la siguiente sesión en base a las 4 últimas sesiones.


----------------------------------------------------------------------------------------------
##### 26.11.2018 ->commit:"Score 0.99"

- [x] Tengo un score del 99.5% lo que me "chirría" bastante. Revisar los datos por si hay algo que se me está escapando.


----------------------------------------------------------------------------------------------
##### 24.11.2018 ->commit:"Incluyo indicadores y patrones de vela"

- [x] Incluyo indicadores técnicos de la librería TA-Lib (todos los que son viables)
- [x] Incluyo todos los patrones de velas que aportan información válida
- [!] El resultado del Algoritmo genético no ha sido como esperaba. Tengo un score de un 50%. Tengo que replantear los 'features' de entrada.

----------------------------------------------------------------------------------------------
##### 16.11.2018 ->commit:"Score 0.99"

- [x] Implemento clase con Algoritmo genético basado en la librería 'deap' para realizar una selección de los mejores hiper-parámetros de la red.


----------------------------------------------------------------------------------------------
##### 6.11.2018 ->commit:"Score 0.99"

- [x] Agrupo todas las funcionalidades del notebook de trabajo en el módulo PredictiveNet.py


----------------------------------------------------------------------------------------------
##### 5.11.2018 ->commit:"Resumen último mes"

- [x] Incluyo un resumen de todos los commits hechos desde el día 19 de octubre:
- [x] Mas pruebas con diferentes configuraciones
- [x] Actualizo pesos nosuffle_bs128
- [x] Actualizado test-predicción. Verificando opción nosuffle_bs128
- [x] Corregida inserción de row en dataframe
- [x] Añadiendo filas al df
- [x] Verificando test recursivo
- [x] Probar sin shuffle, mayor batch_size. Implementar prueba recursiva
- [x] Incluyo archivo test_fourier.csv
- [x] Actualizado walk-forward. Probar con función seno para descartar algún fallo oculto en los datos.
- [x] Comentando Fintech_LSTM. De momento acc=0.68
- [x] Actualizo proceso de entrenamiento
- [x] Probando fintech_test con nuevos features
- [x] Trabajando en fintech_tester
- [x] Probando con fourier multivariate
- [x] Incluyo csv eurusd en M30

