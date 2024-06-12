# FLTrust

`Inclúe a implementación sobre os datos MNIST, DIGIT5 e ROBOTS.`

- MNIST: Os clientes bizantinos executan ataques dirixidos ou ao modelo
- DIGIT5 e ROBOTS: O cliente bizantino traballa datos diferentes que os outros 4 clientes.

Explicación básica dos arquivos:
1. arquivos: Xestiona a modificación dos arquivos de datos
2. bizantine.py: Contén as implementacións dos algoritmos de ataque (MNIST)
3. config.py: Contén as configuracións do experimento
4. datos.py: Contén as funcións de carga e división dos datos
5. debuxar.py: Contén a implementación das funcións de debuxo (scores, precisión)
6. fltrust.py: Conteñen a implementación do modelo FLTrust
7. main.py: Configura e executa o experimento
8. methods.py: Contén a implementación dos métodos de adestramento


Arquivos comúns:
- aggregation.py: Contén a implementación da función de agregación ponderada
- calculos_FLTrust.py: Contén a implementación do cálculo das puntuacións de confianza
- evaluate.py: Contén a implementación das funcións de avaliación
- graficas.py: Contén a implementación das funcións de debuxo (scores, precisión)
- rede.py: Contén as clases das redes empregadas