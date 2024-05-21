Cambios respecto á rama FLDetector:

- Cámbiase a estrutura neuronal da rede CNN, tratando de replicar a arquitectura descrita no paper
- Elimínase a acumulación de gradientes entre clientes, cada un envía o seu gradiente, sen interferencias, ao servidor
- A actualización do modelo no servidor faise mediante a mediana dos gradientes recibidos
- O learning rate pasa a ser de 1.0

Estrutura do repositorio:

- 01_Entreno_Simple: Contén o código para executar un adestramento simple, sen scores nin detección de ataques
- 02_Entreno_Detector: Contén o código para executar un adestramento completo (FLDetector)
  Ambos directorios conteñen 2 arquivos: main (comezo) e mnist_CNN (implementación do algoritmo)

Resto de arquivos:

1. arquivos: Xestiona a modificación dos arquivos de datos
2. bizantine.py: Contén as implementacións dos algoritmos de ataque
3. clases_redes.py: Contén a clase da rede CNN empregada
4. config.py: Contén as configuracións do experimento (parsea os argumentos)
5. debuxar_scores.py: Contén a implementación das funcións de debuxo (scores, precisión e clusterización)
6. detection.py: Contén a implementación das funcións de detección de clientes maliciosos:

- 6.1. detection1: "Comproba" se existe un ataque
- 6.2. detection2: Clusteriza en 2 clases devolve a lista de clientes maliciosos

7. evaluate.py: Contén funcións auxiliares para avaliar o rendemento da rede
8. lbfgs.py: Contén as implementacións da función de optimización L-BFGS (clave para o funcionamento do algoritmo)
9. np_aggregation.py: Contén a implementación dos diferentes métodos de agregación Federados

ARGUMENTOS EXEMPLO PARA MAIN.PY:
--byz_type full_mean_attack --home_path ProbaFLDetector --nepochs 200 --tipo_exec loop

Desta maneira escóllese 'loop' como tipo de execución (unha vez eliminados os clientes, continúa co adestramento)