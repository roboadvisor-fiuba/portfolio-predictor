# portfolio-predictor

Para usar zipline por primera vez, hay que descargar un dataset de testing. Para ello, ejecutar:
'zipline ingest -b quandl'

Para testear con zipline, ejecutar desde /models/zipline:
'zipline run -f evaluation.py --start 2014-1-1 --end 2018-1-1 -o dma.pickle --no-benchmark'

Para entrenar un modelo y generar un pickle (por el momento solo linear regression), ejecutar desde el directorio /models el comando:
'python3 train_model.py'