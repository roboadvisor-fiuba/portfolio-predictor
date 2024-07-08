- cargar zipline bundle para ejecutar notebook de zipline -> ingest tira error por diferencia de data con el calendario (preguntar a NICO)

- terminar backend (endpoint para estadisticas)
    -> prefuntar a nico que estuvo con esto
    - aplicar estrategia sobre la prediccion
    - almacenar prediccion en una db
    - el endpoint hace query de esa db

- random forest no tiene cross validation ni metricas! eso esta en los siguientes notebooks del cap 11 


- ver como analizar prediccion del modelo para estrategia

backend:
- devuelve las predicciones con el modelo cargado
- carga el modelo 
    - lo descarga de un link (si viene por parametro)
    - carga el dump
    - aplica la estrategia sobre el modelo (hardcodeada o embedded en el modelo?)

todo:
- estrategia sobre la prediccion del modelo
- endpoint para graficos y metricas de performance


conseguir un flujo completo con el modelo entrenado de random forest
recien despues, entrenar otros modelos para comparar.