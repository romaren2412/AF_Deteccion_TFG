Funcionamento básico do FLDetector do artigo. Cuasi-réplica do repositorio orixinal, engádese ademáis o filtro 'silueta' como confianza para eliminar os clientes detectados como maliciosos.

[En caso de querer imitar o algoritmo FLDetector orixinal, establecer o argumento 'silueta' a 0.0]

ARGUMENTOS EXEMPLO PARA MAIN.PY:

--byz_type full_mean_attack --aggregation simple_mean --home_path ProbaFLDetector --silhouette 0.8 --nepochs 200 --tipo_exec loop