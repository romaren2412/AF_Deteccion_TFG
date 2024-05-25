# FLARE: Federated LAtent Space REpresentations

Explicación básica dos arquivos:
1. arquivos: Xestiona a modificación dos arquivos de datos
2. bizantine.py: Contén as implementacións dos algoritmos de ataque
3. config.py: Contén as configuracións do experimento (parsea os argumentos)
4. datos.py: Contén as funcións de carga e división dos datos
5. debuxar.py: Contén a implementación das funcións de debuxo (scores, precisión)
6. flare.py: Contén a implementación do modelo FLARE
7. main.py: Configura e executa o experimento
8. methods.py: Contén a implementación dos métodos de adestramento


Arquivos comúns:
- aggregation.py: Contén a implementación dos diferentes métodos de agregación Federados
- calculos_FLARE.py: Contén a implementación das funcións de cálculo de puntuacións Flare
- clases_redes.py: Contén as clases das redes empregadas
- evaluate.py: Contén a implementación das funcións de avaliación