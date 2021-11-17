# Fraud detection

## Notebook de entrenamiento 

El notebook <code>fraud_detection.ipynb </code> contiene en pipeline completo del modelo desde la ingestación hasta la validación de resultados. Asimismo, contiene anotaciones sobre cada paso ejecutado

## Ejecución de python para producción

El archivo <code> main.py </code> es aquel que contiene todo el código para predecir la probabilidad de fraude de una transacción. 

Para ejecutarlo, instalar los requirements:

<code> pip install -r requirements.txt </code>

Posterior a ello ejecutar el python:

<code> python main.py 'archivo a procesar' </code>. Por ejemplo: <code> python main.py prueba.csv </code>.

Este genera un archivo output con 3 columnas:

* User_ID
* Probabilidad de fraude
* Decil asignado

Se adjunta imagen de prueba de ejecución de archivo en producción en servidor local: Prueba_prod.PNG

## Next Steps

* Entrenar con más data para mejorar métricas del modelo
* Explorar otras variables como input al modelo
* Usar técnicas de optimización como grid search o bayesian optimization para encontrar mejorar hiperparámetros

## Insight finales

El objetivo de este sistema de detección de fraude es encontrar de manera oportuna las tarjetas fraudulentas y/o brindar tarjetas que se puedan priorizar en su investigación para evitar quejas de clientes y los investigadores puedan trabajar de manera efectiva
