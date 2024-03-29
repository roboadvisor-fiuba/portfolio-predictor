# portfolio-predictor

Para ejecutar zipline:
'zipline run -f <test_file> --start 2014-1-1 --end 2018-1-1 -o dma.pickle --no-benchmark'

## Instrucciones para levantar el ambiente con Conda

1) Instalar Anaconda o miniconda desde https://github.com/conda-forge/miniforge. Asegurarse de marcar para se agregue el PATH a las variables de entorno. El instalador permite utilizar el gestor conda o mamba. Mamba es más ligero a la hora de descargar paquetes y resolver dependencias pero el uso es indistinto.
2) Abrir una terminal o el Prompt de Miniforge recién instalado. Asegurarse de estar en el env base ejecutando ```mamba env list```. solo debería estar Base.
3) Clonar el repositorio portfolio-predictor a una carpeta local y abrirlo desde VSCode.
4) Desde la terminal o Prompt de Miniforge se debe crear el ambiente virtual. Ubicar el archivo ```portfolio_predictor_env.yml``` en el repo (moverlo de ser necesario), y ejecutar el comando: ```mamba env create -f portfolio_predictor_env.yml```. Esto puede tomar unos minutos ya que descargará los paquetes y creará el ambiente. Se puede ingresar manualmente al ambiente ejecutando ```mamba activate portfolio-predictor``` y ```mamba deactivate``` para salir.
5) Desde VSCode, abrir la paleta superior y buscar ">Python: Select Interpreter". Seleccionar el que diga: Python...('portfolio-predictor'). Una vez realizado esto ya no deberían haber warnings por paquetes no encontrados.
6) Desde una terminal entrar a ```/app``` y ejecutar ```python -m flask run``` para levantar el server en localhost.