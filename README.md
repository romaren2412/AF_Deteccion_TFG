Estrutura da rama do repositorio:
- Adáptase a estrutura ás diferentes probas feitas ata o momento
  - `MNIST`: Código case orixinal do repositorio orixinal de GitHub
  - `DigitFive`: Código adaptado para o dataset `DigitFive`
    - `D0_MNISTM_Ataques`: Ataques untargeted e targeted sobre `MNISTM`
    - `D1_CincoBases`: Execución de 5 clientes con diferentes bases de datos (abocada ao fracaso)
    - `D2_CatroBases_Adicional`: Execución de 5 clientes, 4 comparten base de datos e 1 ten unha base de datos diferente
  - `Robots`: Código adaptado para o dataset dos datos do robot.

  En xeral, cada directorio contén 2 arquivos: main (comezo) e mnist_CNN (implementación do algoritmo)

Explicación básica dos arquivos:
1. arquivos: Xestiona a modificación dos arquivos de datos
2. bizantine.py: Contén as implementacións dos algoritmos de ataque
3. redes.py: Contén a clase da rede CNN empregada
4. config.py: Contén as configuracións do experimento (parsea os argumentos)
5. debuxar.py: Contén a implementación das funcións de debuxo (scores, precisión e clusterización)


Arquivos comúns:
- detection.py: Contén a implementación das funcións de detección de clientes maliciosos:
  - detection1: "Comproba" se existe un ataque
  - detection2: Clusteriza en 2 clases devolve a lista de clientes maliciosos
- evaluate.py: Contén funcións auxiliares para avaliar o rendemento da rede
- lbfgs.py: Contén as implementacións da función de optimización L-BFGS (clave para o funcionamento do algoritmo)
- np_aggregation.py: Contén a implementación dos diferentes métodos de agregación Federados


ARGUMENTOS EXEMPLO:
--byz_type full_mean_attack --home_path ProbaFLDetector --nepochs 200 --tipo_exec loop

Desta maneira escóllese 'loop' como tipo de execución (unha vez eliminados os clientes, continúa co adestramento)