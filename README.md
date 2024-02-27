Funcionamento básico do FLDetector do artigo. Cuasi-réplica do repositorio orixinal, engádese ademáis o filtro 'silueta' como confianza para eliminar os clientes detectados como maliciosos.

[En caso de querer imitar o algoritmo FLDetector orixinal, establecer o argumento 'silueta' a 0.0]

1. bizantine.py: Contén as implementacións dos algoritmos de ataque
2. clases_redes.py: Contén a clase da rede CNN empregada
3. detection.py: Contén a implementación das funcións de detección de clientes maliciosos:
3.1. detection1: "Comproba" se existe un ataque
3.2. detection2: Clusteriza en 2 clases devolve a lista de clientes maliciosos
4. evaluate.py: Contén funcións auxiliares para avaliar o rendemento da rede
5. lbfgs.py: Contén as implementacións da función de optimización L-BFGS (clave para o funcionamento do algoritmo)
6. main.py: Arquivo de execución
7. mnist_CNN.py: Contén a implementación do algoritmo FLDetector (función fl_detector é a clave)
8. np_aggregation.py: Contén a implementación dos diferentes métodos de agregación Federados



ARGUMENTOS EXEMPLO PARA MAIN.PY:
--byz_type full_mean_attack --aggregation simple_mean --home_path ProbaFLDetector --nepochs 200 --tipo_exec loop

Desta maneira escóllese simple_mean (FedAvg) como agregación, e 'loop' como tipo de execución (unha vez eliminados os clientes, continúa co adestramento)