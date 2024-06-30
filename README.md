# FLTrust

Código da sección 4.2.4 no contexto FLTrust
  - `MNIST`: Os clientes bizantinos executan ataques dirixidos ou ao modelo
  - `DigitFive` e `TurtleBot`: O cliente bizantino traballa datos diferentes que os outros 4 clientes.
  - `TurtleBot`: O cliente bizantino traballa datos diferentes que os outros 4 clientes.

En xeral, cada directorio contén 2 arquivos: main (comezo) e fltrust_{dataset} (implementación do algoritmo)

Explicación básica dos arquivos:
1. arquivos: Xestiona a modificación dos arquivos de datos
2. byzantine.py: Contén as implementacións dos algoritmos de ataque a MNIST
3. config.py: Contén as configuracións dos experimentos (parsea os argumentos)
4. datos.py: Xestiona o reparto dos datos de adestramento entre os clientes
5. methods.py: Contén as funcións de adestramento


Arquivos comúns:
- aggregation.py: Contén a implementación dos diferentes métodos de agregación Federados
- calculos_FLARE.py: Contén a implementación das funcións de cálculo de puntuacións Flare
- evaluate.py: Contén a implementación das funcións de avaliación
- redes.py: Contén as clases das redes empregadas
