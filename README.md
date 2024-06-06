# Métodos de agregación robustos: Krum, Median e Trim
Rama do repositorio correspondente ao estudo dos métodos de agregación robustos Krum, Median e Trim.
Executa tanto os ataques simples (mean e backdoor), como os ataques feitos para atacar os métodos robustos.

Explicación básica dos arquivos:
1. arquivos.py: Xestiona a modificación dos arquivos de datos
2. byzantine.py: Contén as implementacións dos algoritmos de ataque
3. config.py: Contén as configuracións do experimento (parsea os argumentos)
4. datos.py: Prepara e reparte os datos entre os clientes
5. evaluate.py: Contén funcións auxiliares para avaliar o rendemento da rede
6. main.py: Inicia o experimento
7. mnist.py: Contén o bucle principal do experimento
8. np_aggregation.py: Contén a implementación dos diferentes métodos de agregación Federados
9. rede.py: Contén a clase da rede CNN empregada