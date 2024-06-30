# FLDetector

Código da sección 4.2 no contexto FLDetector
  - `MNIST`: (Sección 4.2.2 e 4.2.3)
  - `DigitFive`: (Sección 4.2.4) - Execución de 5 clientes, 4 comparten base de datos e 1 ten unha base de datos diferente
  - `TurtleBot`: (Sección 4.2.4) - Código adaptado para o dataset dos datos do robot.

En xeral, cada directorio contén 2 arquivos: main (comezo) e fld_{dataset} (implementación do algoritmo)

Explicación básica dos arquivos:
1. arquivos: Xestiona a modificación dos arquivos de datos
2. bizantine.py: Contén as implementacións dos algoritmos de ataque a MNIST
3. config.py: Contén as configuracións dos experimentos (parsea os argumentos)
4. datos.py: Xestiona o reparto dos datos de adestramento entre os clientes
5. methods.py: Contén as funcións de adestramento


Arquivos comúns:
- aggregation.py: Contén a implementación dos diferentes métodos de agregación
- detection.py: Contén a implementación das funcións de detección de clientes maliciosos
- evaluate.py: Contén funcións auxiliares para avaliar o rendemento da rede
- lbfgs.py: Contén as implementacións da función de optimización L-BFGS
- redes.py: Contén a clase das redes neuronais empregadas