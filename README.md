# 4.1 Ataques bizantinos
Rama do repositorio correspondente á sección 4.1:
- Rendemento de FedAvg en presenza de ataques
- Impacto da aplicación de métodos robustos de agregación: Krum, Median e Trimmed-Mean
- Debilidades específicas dos métodos robustos (Ataques deseñados)
- Combinación de métodos robustos: Multikrum, Bulyan e Bultibulyan.

Explicación básica dos arquivos:
1. aggregation.py: Contén a implementación dos diferentes métodos de agregación federados
2. arquivos.py: Xestiona a modificación dos arquivos de datos
3. byzantine.py: Contén as implementacións dos algoritmos de ataque 
4. config.py: Contén as configuracións do experimento (parsea os argumentos)
5. datos.py: Prepara e reparte os datos entre os clientes 
6. evaluate.py: Contén funcións auxiliares para avaliar o rendemento da rede 
7. main.py: Inicia o experimento 
8. mnist.py: Contén o bucle principal do experimento
9. rede.py: Contén a clase da rede CNN empregada